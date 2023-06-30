"""Top level manager and utility code to report annotation problems and report
stats.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Tuple, Set, List, Any, Callable, Iterable, Dict
from dataclasses import dataclass, field
import logging
import sys
from io import TextIOBase
from pathlib import Path
import numpy as np
import pandas as pd
from zensols.config import ConfigFactory
from zensols.persist import persisted, ReadOnlyStash, FactoryStash
from zensols.mimic import Note, DischargeSummaryNote, Corpus
from zensols.datdesc import DataFrameDescriber, DataDescriber
from . import (
    Issue, IssueContainer, NoteMatchSetStash, MatchBuilder, MatchedAnnotationSet
)

logger = logging.getLogger(__name__)


@dataclass
class MatchAnnotationSetFactoryStash(ReadOnlyStash):
    """Creates instances of :class:`.MatchedAnnotationSet`.

    """
    config_factory: ConfigFactory = field()
    """Used to create instances of :class:`.MatchBuilder`."""

    note_match_set_stash: NoteMatchSetStash = field()
    """The stash of :class:`.NoteAntecedentSet` instances."""

    match_builder_name: str = field()
    """The name of the section with the :class:`.MatchBuilder` instance
    definition.

    """
    def create_match_builder(self, hadm_id: str) -> MatchBuilder:
        """Create a match builder from the config factory.

        :param hadm_id: the admission ID of the data to match

        """
        return self.config_factory.new_instance(
            self.match_builder_name, hadm_id=hadm_id)

    def load(self, name: str) -> MatchedAnnotationSet:
        """Return the matched annotation set for an admission.

        :param hadm_id: the hospital admission ID

        """
        match_builder: MatchBuilder = self.create_match_builder(name)
        return match_builder.match_annotations

    def keys(self) -> Iterable[str]:
        """The hospital admission IDs of the annotated corpus."""
        return self.note_match_set_stash.keys()

    def exists(self, name: str) -> bool:
        return self.note_match_set_stash.exists(name)


@dataclass
class MatchAnnotationSetStash(FactoryStash):
    """Uses a backing store with the :class:`MatchAnnotationSetFactoryStash` to
    create instances of :class:`.MatchedAnnotationSet`.

    """
    corpus: Corpus = field(default=None)
    """The contains assets to access the MIMIC-III corpus via database."""

    def create_match_builder(self, hadm_id: str) -> MatchBuilder:
        return self.factory.create_match_builder(hadm_id)

    def load(self, name: str) -> MatchedAnnotationSet:
        ma: MatchedAnnotationSet = super().load(name)
        self.factory.create_match_builder(name)._bless_annotation(ma)
        return ma


@dataclass
class AdmissionSimilarityAssessor(IssueContainer):
    """Summarizes statistics and make sense of match data.

    """
    config_factory: ConfigFactory = field(repr=False)
    """Used to create instances of :class:`.MatchBuilder`."""

    match_ann_stash: MatchAnnotationSetStash = field(repr=False)
    """Creates instances of :class:`.MatchedAnnotationSet`."""

    match_builder_name: str = field()
    """The name of the section with the :class:`.MatchBuilder` instance
    definition.

    """
    corpus: Corpus = field()
    """The contains assets to access the MIMIC-III corpus via database."""

    note_match_set_stash: NoteMatchSetStash = field()
    """The stash of :class:`.NoteAntecedentSet` instances."""

    data_dir: Path = field()
    """The directory used to store cached data files."""

    results_dir: Path = field()
    """The directory where results are saved."""

    dataframe_describer_name: str = field()
    """The name of the section that defines the :class:`.DataFrameDescriber`."""

    @property
    def hadm_ids(self) -> Set[str]:
        """The hospital admission IDs of the annotated corpus."""
        return set(self.note_match_set_stash.keys())

    @property
    @persisted('_match_dataframe')
    def match_dataframe(self) -> pd.DataFrame:
        """A dataframe of all matched data entries.  This dataframe is used to
        create statistics for output that attempt to give an idea of the
        copy/paste physicians when writing discharge summaries.

        """
        hadm_ids: Set[str] = self.hadm_ids
        dfs: List[pd.DataFrame] = []
        hadm_id: str
        for hadm_id in hadm_ids:
            mas: MatchedAnnotationSet = self.match_ann_stash[hadm_id]
            df: pd.DataFrame = mas.dataframe
            dfs.append(df)
        return pd.concat(dfs)

    def find_text_in_notes(self, text: str, hadm_id: str):
        """Find ``text`` by iterating through all notes of the admission."""
        notes: Tuple[Note] = self.corpus.hospital_adm_stash[hadm_id]
        for note in sorted(notes, key=lambda n: n.row_id, reverse=True):
            if note.category == DischargeSummaryNote.CATEGORY:
                continue
            if note.text.find(text) > -1:
                print(f'storetime: {note.storetime}')
                note.write()

    def _add_issues(self, issues: List[Issue]):
        hadm_ids: Set[str] = set(self.note_match_set_stash.keys())
        hadm_id: str
        for hadm_id in hadm_ids:
            match_builder: MatchBuilder = self.match_ann_stash.\
                create_match_builder(hadm_id)
            match_builder._add_issues(issues)

    def _create_dataframe_describer(self, df: pd.DataFrame,
                                    name: str, desc: str) -> DataFrameDescriber:
        dd = self.config_factory.new_instance(
            self.dataframe_describer_name, df=df, desc=desc)
        dd.name = name
        return dd

    def _coverage_by_admission(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Get the overlap coverage between the discharge notes an respective
        note antecedents for all notes.

        :param df: use if provided (i.e. subset) of :obj:`match_dataframe`

        :return: the dataframe with each row as an admission

        """
        def col_vals(df: pd.DataFrame, prefix: str):
            """Create the column values for total, matching and portions by
            aggregating across the comment ID.

            Discharge sumamries are aggregated using the max since all values
            originate from the same comment (branches on discharge summary, then
            note antecedent).  Note antecendents are aggregated using sum since
            each value is that match to the respective discharge summary
            comment.

            """
            agg: str = {'ds': 'max', 'ant': 'sum'}[prefix]
            # compute totals used as the divisor of the proportion's quotient
            cols: List[str] = f'{prefix}_tot_tok {prefix}_tot_char'.split()
            dft: pd.DataFrame
            if prefix == 'ds':
                # ds have the same total counts for the entire admission since
                # we assume one discharge summary per admission
                dft = df[cols].drop_duplicates()
                assert len(dft) == 1
                tot_toks, tot_chars = dft.iloc[0].values
            else:
                # ants will have the same totals for each comment
                dft = df.groupby('comment_id')[cols].agg('max')
                tot_toks, tot_chars = dft.sum().values
            # compute matches used as the numerator of the proportion's quotient
            cols = f'{prefix}_tok {prefix}_char'.split()
            dfm: pd.DataFrame = df.groupby('comment_id')[cols].agg(agg)
            # sum on the aggregate: ds max to get unique across comment ID, ant
            # is the sum to count each antecedent
            match_toks, match_chars = dfm.sum()
            cols = f'{prefix}_portion_tok {prefix}_portion_char'.split()
            por_toks, por_chars = match_toks / tot_toks, match_chars / tot_chars
            return (tot_toks, tot_chars,
                    match_toks, match_chars,
                    por_toks, por_chars)

        dfa: pd.DataFrame = self.match_dataframe if df is None else df
        # all the columns in the resulting dataframe
        cols: List[str] = (
            'hadm_id annotator ' +
            'ds_tot_toks ds_tot_chars ' +
            'ds_match_toks ds_match_chars ' +
            'ds_portion_toks ds_portion_chars ' +
            'ant_tot_toks ant_tot_chars ' +
            'ant_match_toks ant_match_chars ' +
            'ant_portion_toks ant_portion_chars').split()
        rows: List[List[Any]] = []
        hadm_id: str
        df: pd.DataFrame
        for hadm_id, df in dfa.groupby('hadm_id'):
            anns: List[str] = df['annotator'].drop_duplicates().to_list()
            assert len(anns) == 1
            row: List[Any] = [hadm_id, anns[0]]
            rows.append(row)
            row.extend(col_vals(df, 'ds'))
            row.extend(col_vals(df, 'ant'))
        df = pd.DataFrame(rows, columns=cols)
        df = df.sort_values('ds_portion_chars', ascending=False)
        return df

    def _coverage_by_group(self, df: pd.DataFrame, by_col: str,
                           index_col_name: str,
                           count_col_name: str,
                           sort_col: str = None) -> pd.DataFrame:
        """Create a coverage dataframe using grouping by a column from
        :obj:`match_dataframe`.

        :param by_col: the column to group on (pandas ``groupby``)

        :param index_col_name: the name of the column output of the groupby
                               column

        :param count_col_name: the name of the column output of the count
                               column

        :param sort_col: if given, sort the dataframe on this column

        """
        score_cols: List[str] = MatchBuilder.SCORE_COLS
        df: pd.DataFrame = self.match_dataframe
        dfc = df.groupby(by_col)[by_col].agg('count').astype(np.int).\
            to_frame().rename(columns={by_col: 'count'})
        dfc.index.name = index_col_name
        rows: List[str] = []
        cols: Tuple[str] = ('ds_match_toks ds_match_chars ' +
                            'ant_match_toks ant_match_chars').split()
        t_cols: Tuple[str] = ('hadm_id ds_tot_toks ds_tot_chars ' +
                              'ant_tot_toks ant_tot_chars').split()
        for group_col, count in dfc.itertuples():
            dff = df[df[by_col] == group_col]
            score_means: float = dff[score_cols].mean()
            dff = self._coverage_by_admission(dff)
            dfd = dff[t_cols].drop_duplicates().drop(columns={'hadm_id'}).sum()
            dff = dff[cols].sum()
            dff = pd.concat((dff, dfd))
            dff[index_col_name] = group_col
            dff[count_col_name] = count
            for col in score_cols:
                dff[col] = score_means[col]
            rows.append(dff)
        df = pd.DataFrame(rows)
        cols = list(df.columns)
        cols = cols[-3:] + cols[:-3]
        df = df[cols]
        for prefix in 'ds ant'.split():
            for suffix in 'toks chars'.split():
                mcol = f'{prefix}_match_{suffix}'
                pcol = f'{prefix}_portion_{suffix}'
                tcol = f'{prefix}_tot_{suffix}'
                df[pcol] = df[mcol] / df[tcol]
        if sort_col is not None:
            df = df.sort_values(sort_col, ascending=False)
        return df

    def _coverage_by_category(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Like :meth:`_coverage_by_admission` but by note category.

        :param df: use if provided (i.e. subset) of :obj:`match_dataframe`

        :return: the dataframe with each row as a note category

        """
        return self._coverage_by_group(
            df, 'ant_cat', 'category', 'note_count', 'ant_portion_chars')

    def _coverage_by_ds_section(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Like :meth:`_coverage_by_admission` but by discharge summary section
        ID.

        :param df: use if provided (i.e. subset) of :obj:`match_dataframe`

        :return: the dataframe with each row as a note section

        """
        return self._coverage_by_group(
            df, 'ds_sec_id', 'section', 'section_count', 'ds_portion_chars')

    def _coverage_by_ant_section(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Like :meth:`_coverage_by_admission` but by note antecedent section
        ID.

        :param df: use if provided (i.e. subset) of :obj:`match_dataframe`

        :return: the dataframe with each row as a note section

        """
        return self._coverage_by_group(
            df, 'ant_sec_id', 'section', 'section_count', 'ant_portion_chars')

    def _corpus_stats(self) -> pd.DataFrame:
        df: pd.DataFrame = self.match_dataframe
        stats: Dict[str, int] = {
            'n_hadm': len(df['hadm_id'].drop_duplicates()),
            'n_ds_notes': len(df['ds_row_id'].drop_duplicates()),
            'n_ant_notes': len(df['ant_row_id'].drop_duplicates()),
            'n_annotators': len(df['annotator'].drop_duplicates()),
            'n_matches': len(df),
        }
        scols: List[str] = ('ds_tot_tok ds_tot_char ant_tot_tok ' +
                            'ds_tot_char ant_tot_char').split()
        stats |= dict(map(lambda c: (c, df[c].sum()), scols))
        stats['n_notes'] = stats['n_ds_notes'] + stats['n_ant_notes']
        stats['n_toks'] = stats['ds_tot_tok'] + stats['ant_tot_tok']
        stats['n_chars'] = stats['ds_tot_char'] + stats['ant_tot_char']
        return pd.DataFrame([stats])

    def _get_dataframe_metadata(self) -> Tuple[str, str, Callable]:
        """Return multiple information and how to create them.  This data is
        used to create the Excel file.

        :return: a tuple of:

            * name, which is used in the sheet name in the Excel file
            * short humn readable description of the dataframe
            * a :class:`Callable`, which creates the dataframe

        """
        return (('Note Category',
                 'Statistics grouped by MIMIC note category',
                 self._coverage_by_category),
                ('Discharge Summary Section ID',
                 'Statistics grouped by discharge summary section ID',
                 self._coverage_by_ds_section),
                ('Note Antecedent Section ID',
                 'Statistics grouped by note antecedent section ID',
                 self._coverage_by_ant_section),
                ('Coverage by Admission',
                 'Statistics grouped by admission (hadm_id)',
                 self._coverage_by_admission),
                ('Corpus Statistics',
                 'Basic corpus statistics',
                 self._corpus_stats),
                ('Match',
                 'All match data summarized in the other sheets',
                 lambda: self.match_dataframe))

    def _get_by_adm_dataframe_describer(self) -> DataFrameDescriber:
        df: pd.DataFrame = self.match_dataframe.\
            groupby('hadm_id')['hadm_id'].agg('count').astype(int).\
            to_frame().rename(columns={'hadm_id': 'notes_by_adm'}).\
            describe().\
            drop(index='50%'.split()).\
            T
        return DataFrameDescriber(
            name='Notes by Admission',
            df=df,
            desc='Note counts per admission',
            meta=(('count', 'Count'),
                  ('mean', 'Mean'),
                  ('std', 'Standard Deviation'),
                  ('min', 'Minimum'),
                  ('max', 'Maximum'),
                  ('25%', '25 Percentile'),
                  ('75%', '75 Percentile')))

    def get_data_describer(self) -> DataDescriber:
        """Return the data and metadata of the matched data."""
        dfds: List[DataFrameDescriber] = []
        meths: Tuple[str, str, Callable] = self._get_dataframe_metadata()
        for name, desc, meth in meths:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'creating dataframe {name}: {desc}')
            df: pd.DataFrame = meth()
            dfd: DataFrameDescriber = self._create_dataframe_describer(
                df=df, name=name, desc=desc)
            dfds.append(dfd)
        dfds.append(self._get_by_adm_dataframe_describer())
        return DataDescriber(tuple(dfds))

    def save_excel(self, output_file: Path = None):
        """Save all statistics data to an Excel file.

        :param output_file: the ``.xlsx`` file to save (extension needs to be
                            added)

        """
        dd: DataDescriber = self.get_data_describer()
        if output_file is None:
            output_file = self.results_dir / 'match.xlsx'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        dd.save_excel(output_file)
        logger.info(f'wrote: {output_file}')

    def save_csv(self, output_dir: Path = None, yaml_dir: Path = None):
        """Save all statistics data to an Excel file.

        :param output_dir: the directrory where the results are saved

        """
        dd: DataDescriber = self.get_data_describer()
        if output_dir is None:
            output_dir = self.results_dir / 'match'
        output_dir.mkdir(parents=True, exist_ok=True)
        dd.save_csv(output_dir)
        if yaml_dir is not None:
            dd.save_yaml(output_dir, yaml_dir)
        logger.info(f'wrote files to: {output_dir}')

    def clear(self):
        """Clear the cached :obj:`match_dataframe` instance.

        """
        self.match_ann_stash.clear()

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        from . import CommentIssue
        self.write_issues(excludes={CommentIssue})
