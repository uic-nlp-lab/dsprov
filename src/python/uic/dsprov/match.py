"""Contains classes for matching dischrage summary antecedents with originating
medical notes.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Iterable, Tuple, List, Dict, Any, Union, ClassVar
from dataclasses import dataclass, field
import logging
import sys
import re
import itertools as it
import textwrap
from io import TextIOBase, StringIO
from pathlib import Path
import pandas as pd
from zensols.config import ConfigFactory
from zensols.persist import ReadOnlyStash, persisted
from zensols.mimic import Note, HospitalAdmission, Corpus
from zensols.nlp import (
    LexicalSpan, FeatureToken, TokenContainer, FeatureSpan, FeatureDocument,
)
from zensols.nlp.score import ScoreContext, ScoreSet, Scorer
from zensols.mimic import Section, NoteFactory
from . import (
    ProvenanceError, MissingNoteError, AdmissionError,
    Annotation, DischargeSummaryAnnotation,
    DischargeSummaryAnnotationStash, NoteAntecedent, NoteAntecedentSet,
    SpanCounts, MatchedNote, MatchedSection, MatchedAnnotation,
    MatchedAnnotationSet, Issue, IssueContainer,
)
from .domain import _MatchResource, AnnotationBase

logger = logging.getLogger(__name__)


@dataclass
class NoteMatcher(object):
    """Antecedents discharge annotations with where they were referenced in other
    notes bounded by the scope of the admission.  The text to look for is
    copied/pasted in the comments by the physicians in the annotated discharge
    summary MS Word document file.

    """
    corpus: Corpus = field()
    """A container class for the resources that access the MIMIC-III corpus."""

    dsanon_stash: DischargeSummaryAnnotationStash = field()
    """A stash of :class:`.DischargeSummaryAnnotation` instances."""

    hadm_id: str = field()
    """The admission ID used to match from discharge summaries to notes."""

    @staticmethod
    def _find_all(text: str, query: str) -> Iterable[LexicalSpan]:
        """Find all instances of ``query`` in ``text``.

        :param text: the text on which to search

        :param query: the substring to find

        :return: an iterable on the antecedents as spans

        """
        start: int = 0
        while True:
            start = text.find(query, start)
            if start == -1:
                break
            end: int = start + len(query)
            if logger.isEnabledFor(logging.DEBUG):
                found_text: str = text[start:end]
                found_text = textwrap.shorten(found_text, 70)
                query_text: str = query
                query_text = textwrap.shorten(query, 70)
                logger.debug(f'found text: <{query_text}> -> <{found_text}>')
            yield LexicalSpan(start, end)
            start = end

    def __call__(self) -> \
            Tuple[DischargeSummaryAnnotation, List[NoteAntecedent]]:
        """Return all antecedents from the discharge notes to the antecedent
        copy/pasted text.

        """
        dsanon: DischargeSummaryAnnotation = self.dsanon_stash[self.hadm_id]
        adm: HospitalAdmission = self.corpus.hospital_adm_stash[self.hadm_id]
        antecedents: List[NoteAntecedent] = []
        anon: Annotation
        for anon in dsanon.text_match_annotations:
            note_id: str = anon.note_id
            targ: str = anon.note_text
            if note_id not in adm:
                raise MissingNoteError(self.hadm_id, note_id, targ)
            else:
                note: Note = adm.notes_by_id[note_id]
                antecedents.append(NoteAntecedent(
                    annotation=anon,
                    spans=tuple(self._find_all(note.text, targ))))
        return dsanon, antecedents


@dataclass
class NoteMatchSetStash(ReadOnlyStash):
    """Manages instances of :class:`.NoteAntecedent`.

    """
    config_factory: ConfigFactory = field()
    """The factory used to create matcher resources."""

    dsanon_stash: DischargeSummaryAnnotationStash = field()
    """A stash of :class:`.DischargeSummaryAnnotation` instances."""

    matcher_name: str = field()
    """The section name of the :class:`.NoteMatcher` instance definition."""

    def load(self, hadm_id: str) -> NoteAntecedentSet:
        if self.exists(hadm_id):
            matcher: NoteMatcher = self.config_factory.new_instance(
                self.matcher_name, hadm_id=hadm_id)
            dsanon: DischargeSummaryAnnotation
            antecedents: List[NoteAntecedent]
            dsanon, antecedents = matcher()
            if len(antecedents) == 0:
                logger.warning(f'no antencedents found for adm {hadm_id}')
            aset = NoteAntecedentSet(
                discharge_summary_annotation=dsanon,
                antecedents=antecedents)
            return aset

    def keys(self) -> Iterable[str]:
        return self.dsanon_stash.keys()

    def exists(self, hadm_id: str) -> bool:
        return self.dsanon_stash.exists(hadm_id)


@dataclass
class _Row(object):
    data: List[Union[str, float, int]] = field(default_factory=list)
    tokens: TokenContainer = field(default=None)

    def clone(self) -> _Row:
        return self.__class__(
            data=list(self.data),
            tokens=self.tokens)

    def append(self, x):
        self.data.append(x)

    def extend(self, x):
        self.data.extend(x)

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class MatchBuilder(IssueContainer):
    """Builds instances of :class:`.MatchedAnnotationSet` and a
    :class:`.pandas.DataFrame` from them.

    """
    SCORE_COLS: ClassVar[List[str]] = (
        'levenshtein bertscore rouge1_f_score rouge2_f_score rougeL_f_score ' +
        'bleu semeval_partial_f_score').split()
    """"""

    corpus: Corpus = field(repr=False)
    """A container class for the resources that access the MIMIC-III corpus."""

    mimic_note_factory: NoteFactory = field(repr=False)
    """The factory that creates :class:`.Note` for hopsital admissions."""

    note_match_set_stash: NoteMatchSetStash = field()
    """The stash of :class:`.NoteAntecedentSet` instances."""

    meta_path: Path = field()
    """The path to the file containing the column metadata for
    :obj:`dataframe`.

    """
    scorer: Scorer = field()
    """Used to add similiarty metrics between spans."""

    hadm_id: str = field()
    """The admission ID used to build the the matched note set (see class
    docs).

    """
    def __post_init__(self):
        self._match_resource = _MatchResource(
            corpus=self.corpus,
            mimic_note_factory=self.mimic_note_factory)

    def _narrow_matched_text(self, ms: MatchedSection, note_text: str):
        if note_text is not None:
            spans: List[LexicalSpan] = list(ms.header_spans)
            if ms.body_span is not None:
                spans.append(ms.body_span)
            spans.sort()
            tio = StringIO()
            for span in spans:
                tio.write(note_text[span.begin:span.end])
            ms.matched_text = tio.getvalue()

    def _get_overlapping(self, ant: NoteAntecedent, note: Note,
                         span: LexicalSpan, text_attr: str) -> MatchedNote:
        doc: FeatureDocument = note.doc
        tokens: Tuple[FeatureToken] = tuple(
            doc.get_overlapping_tokens(span, inclusive=False))
        n_tokens = sum(1 for _ in tokens)
        n_ws_tokens = sum(1 for _ in re.split(r'\s+', ant.annotation.text))
        n_tokens = sum(1 for _ in tokens)
        note_text: str = note.text[span.begin:span.end]
        if logger.isEnabledFor(logging.DEBUG):
            ann_text: str = ant.annotation.text
            if 0:
                note_text = textwrap.shorten(note_text, 40)
                ann_text = textwrap.shorten(ann_text, 40)
            logger.debug(f'hadm: {self.hadm_id}, note id: {note.row_id}, ' +
                         f'note type: {note.__class__.__name__}, ' +
                         f'annotation text: <{ann_text}>, ' +
                         f'to find: <{note_text}>')
            logger.debug(
                f'ds tokens: ws={n_ws_tokens}, parsed={n_tokens}')
            if 0:
                logger.debug('|'.join(
                    map(lambda t: t.text,
                        doc.get_overlapping_tokens(span, inclusive=False))))
                logger.debug('|'.join(re.split(r'\s+', ant.annotation.text)))
        sections: List[Section] = []
        sec: Section
        for sec in note.sections.values():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'searching for {span} in section {sec}, ' +
                             f'body span: {sec.body_span}')
            if sec.body_span.overlaps_with(span):
                htoks: List[FeatureToken] = []
                hspan: LexicalSpan = None
                bspan: LexicalSpan = LexicalSpan.EMPTY_SPAN
                hspans: Tuple[LexicalSpan] = []
                for hspan in sec.header_spans:
                    hspan = span.narrow(hspan)
                    if hspan is not None:
                        hspans.append(hspan)
                        htoks.extend(doc.get_overlapping_tokens(hspan, inclusive=False))
                if sec.body_span is not None:
                    bspan = span.narrow(sec.body_span)
                name: str = sec.name
                if name == 'hpi':
                    name = 'history-of-present-illness'
                htspan = FeatureSpan(tuple(htoks))
                htspan.strip()
                ms = MatchedSection(
                    section_id=name,
                    text=note_text,
                    header_spans=hspans,
                    header_tokens=htspan,
                    body_span=bspan)
                self._narrow_matched_text(ms, note.text)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'matched: {ms}')
                sections.append(ms)
        if len(sections) == 0:
            if logger.isEnabledFor(logging.WARNING):
                txt: str = textwrap.shorten(note_text, 40)
                logger.warning(
                    f'no sections with text <{txt}> found for {note}')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'matched sections ({len(sections)}): {sections}')
        tspan = FeatureSpan(tokens)
        tspan.strip()
        note = MatchedNote(
            hadm_id=self.hadm_id,
            row_id=note.row_id,
            cid=ant.annotation.cid,
            note_category=note.category,
            text=getattr(ant.annotation, text_attr),
            span=tspan.lexspan,
            sections=sections)
        return note

    def _discharge(self, ant: NoteAntecedent, ds_note: Note) -> MatchedNote:
        ds_span: LexicalSpan = ant.annotation.span
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'find matches in discharge note {ds_note} ' +
                         f'({ds_note.__class__.__name__}) in {ant} ' +
                         f'({ant.__class__.__name__})')
        return self._get_overlapping(ant, ds_note, ds_span, 'note_text')

    def _source(self, ant: NoteAntecedent, note: Note) -> Tuple[MatchedNote]:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'find matches in antecident {note} '
                         f'({note.__class__.__name__})')
        return tuple(map(lambda s: self._get_overlapping(ant, note, s, 'text'),
                         ant.spans))

    def _create_matched_anonation(self, aset: NoteAntecedentSet,
                                  ant: NoteAntecedent) -> MatchedAnnotation:
        ant_row_id: str = ant.annotation.note_id
        ds_note: Note = aset.discharge_summary
        src_note: Note = aset.get_source_note(ant_row_id)
        try:
            ds_match: MatchedNote = self._discharge(ant, ds_note)
            if len(ds_match.sections) == 0:
                # this happens for non-human annotated sections in the discharge
                # summary; but could happenn for predicted discharge summaries
                ds_row_id: str = ds_note.row_id
                ds_note = ant.get_source_note(ds_row_id, note_regex=False)
                ds_match = self._discharge(ant, ds_note)
        except Exception as e:
            raise ProvenanceError(
                f'Can not create discharge from antecedent {ant} ' +
                f'on note: {ds_note}') from e
        sources: Tuple[MatchedNote] = self._source(ant, src_note)
        if any(filter(lambda s: len(s.sections) == 0, sources)):
            hadm_id: str = aset.discharge_summary_annotation.hadm_id
            logger.warning('note antecedent has at least one source with ' +
                           f'no matches for adm={hadm_id}--trying non-regex ' +
                           f'instance: {src_note}')
            src_note = aset.get_source_note(ant_row_id, note_regex=False)
            sources = self._source(ant, src_note)
            if len(sources) == 0:
                raise AdmissionError(
                    hadm_id,
                    'Note antecedent has at least one source with ' +
                    f'no matches for note: {src_note}')
        # use 'issues' action instead of raise an error when 0 sources found
        return MatchedAnnotation(
            discharge_summary=ds_match,
            note_antecedents=sources,
            annotator=aset.discharge_summary_annotation.annotator)

    def _bless_annotation(self, anon: AnnotationBase):
        anon._set_resource(self._match_resource)

    @property
    @persisted('_match_annotations')
    def match_annotations(self) -> MatchedAnnotationSet:
        """Match discharge summary notes with note antecedents.  It does this by
        matching across the discahrge note section with the note antecedent
        section.  The discharge summary section is matched using the selected
        text in the discharge summary MS Word.  The discharge summary *must* be
        annotated.

        It matches the discharge summary by iterating over each section to find
        the text span, which it might not if the discahrge summary was annotated
        in such a way the text doesn't overlap or the text was not annotated as
        a section.

        It matches the note antecedent by iterating over each section like the
        discharge summary, which it might not find for the same annotation
        reasons.  It might also be because there are no annotations for the
        note, but the note is a category with the hand-crafted regular
        expressions that don't match the text in the span.

        In the case of missing matches on the note antecedent, a "vanilla"
        :class:`~zensols.mimic.note.Note` is created with the singleton
        "default" section, and then rematched.  This might happen with MedSecId
        automatically annotated notes, but should happen less.

        :raises ProvenanceError: if the MS Word document selected text is not
                                 found in any section of the discharge summary

        """
        aset: NoteAntecedentSet = self.note_match_set_stash.load(self.hadm_id)
        manons: List[MatchedAnnotation] = []
        self._bless_annotation(aset)
        ant: NoteAntecedent
        for ant in aset.antecedents:
            manons.append(self._create_matched_anonation(aset, ant))
        mas = MatchedAnnotationSet(
            hadm_id=self.hadm_id,
            matches=manons,
            dataframe=None)
        self._bless_annotation(mas)
        self._populate_dataframe(mas)
        return mas

    def _add_issues(self, issues: List[Issue]):
        aset: NoteAntecedentSet = self.note_match_set_stash.load(self.hadm_id)
        mas: MatchedAnnotationSet = self.match_annotations
        self._bless_annotation(aset)
        self._bless_annotation(mas)
        aset._add_issues(issues)
        mas._add_issues(issues)

    @persisted('_columns')
    def _get_columns(self) -> Tuple[str, str]:
        df = pd.read_csv(self.meta_path)
        df = df[df['type'] == 'base']
        return tuple(df.itertuples(index=False, name=None))

    def _create_ds_rows(self, mas: MatchedAnnotationSet,
                        match: MatchedAnnotation, ds_row: Tuple) -> \
            List[_Row]:
        ds_note: MatchedNote = match.discharge_summary
        assert type(ds_note) == MatchedNote
        if len(ds_note.sections) == 0:
            match.write()
            # this happens when no sections matched
            raise AdmissionError(
                hadm_id=mas.hadm_id,
                msg=(f'No discharge summary sections in note {ds_note} ' +
                     f'for: {textwrap.shorten(str(match), 60)}'))
        else:
            rows: List[_Row] = []
            ds_sec: MatchedSection
            # use only the first as discharge summary selected text that spans
            # multiple sections are duplicates (see :meth:`_get_overlapping`)
            for ds_sec in it.islice(ds_note.sections, 1):
                row = _Row()
                row.tokens = ds_sec.tokens
                row.extend(ds_row)
                rows.append(row)
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f'single discharge counts ({ds_note.hadm_id})' +
                                f': {ds_sec.counts}')
                row.append(match.annotator)
                row.append(ds_note.cid)
                row.append(ds_sec.matched_text)
                row.extend(ds_sec.counts)
                row.append(ds_sec.section_id)
        if len(match.note_antecedents) == 0:
            # leave this as a warning; otherwise the (better) mismatch issue
            # won't get logged, which has more info
            logger.warning(f'no matched antecedents for adm: {self.hadm_id}')
            #raise AdmissionError(self.hadm_id, 'No matched antecedents')
        return rows

    def _score(self, ds: TokenContainer, ant: TokenContainer) -> List[float]:
        ss: ScoreSet = self.scorer.score(ScoreContext(
            pairs=[[ds, ant]]))
        df: pd.DataFrame = ss.as_dataframe()
        assert len(df) == 1
        return df[self.SCORE_COLS].iloc[0].to_list()

    def _add_ant_row(self, adm: HospitalAdmission, mas: MatchedAnnotationSet,
                     ant_match: MatchedNote, rowc: _Row, rows: List[_Row]):
        def tnorm(s: str) -> str:
            return '|'.join(re.split(r'\s+', s.strip()))

        assert type(ant_match) == MatchedNote
        rid: int = ant_match.row_id
        note: Note = adm[rid]
        nac_counts: SpanCounts = mas.note_antecedent_counts[rid]
        ant_sec_row: _Row = rowc.clone()
        ant_sec_row.append(str(rid))
        ant_sec_row.append(note.category)
        ant_sec_row.extend(nac_counts)
        if len(ant_match.sections) == 0:
            raise ProvenanceError(
                f'Not sections for antecedent match: {ant_match}')
        ant_sec: MatchedSection
        for ant_sec in ant_match.sections:
            assert type(ant_sec) == MatchedSection
            scores: List[float] = self._score(rowc.tokens, ant_sec.tokens)
            ams_row: _Row = ant_sec_row.clone()
            rows.append(ams_row)
            ams_row.append(ant_sec.matched_text)
            ams_row.append(ant_sec.section_id)
            ams_row.extend(ant_sec.counts)
            ams_row.extend(scores)

    def _populate_dataframe(self, mas: MatchedAnnotationSet) -> pd.DataFrame:
        adm: HospitalAdmission = self.corpus.hospital_adm_stash[mas.hadm_id]
        # get the columns from the metadata
        cols: Tuple[str] = tuple(map(lambda x: x[0], self._get_columns()))
        ds_all: SpanCounts = mas.discharge_summary_counts
        rows: List[_Row] = []
        ds_row: Tuple = (mas.hadm_id, str(mas.discharge_summary_row_id),
                         ds_all.tokens, ds_all.chars)
        match: MatchedAnnotation
        for match in mas.matches:
            ds_rows: List[_Row] = self._create_ds_rows(mas, match, ds_row)
            row: _Row
            for row in ds_rows:
                if row is None:
                    continue
                ant_match: MatchedNote
                for ant_match in match.note_antecedents:
                    self._add_ant_row(adm, mas, ant_match, row, rows)
        row_data = list(map(lambda r: r.data, rows))
        mas.dataframe = pd.DataFrame(row_data, columns=cols)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_object(self.match_annotations, depth, writer)

    def _from_dictable(self, *args, **kwargs) -> Dict[str, Any]:
        return self.match_annotations.asdict()
