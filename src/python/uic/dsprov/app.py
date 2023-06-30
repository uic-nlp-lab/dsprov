"""Contains classes for creating and accessing discharge summary *provenience
of data* annotations.

"""
__author__ = 'Paul Landes'

from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import sys
import itertools as it
from pathlib import Path
import pandas as pd
from zensols.config import ConfigFactory
from zensols.persist import Stash
from zensols.cli import LogConfigurator
from zensols.datdesc import DataDescriber, DataFrameDescriber
from zensols.mimic import HospitalAdmission, Corpus
from zensols.mimicsid import AnnotationResource
from . import (
    ProvenanceWriter, DischargeSummaryAnnotation,
    DischargeSummaryAnnotationStash, AdmissionSimilarityAssessor,
    Issue, CommentIssue,
)
from .reader import ProvenanceReader
from . import MatchedAnnotationSet, AnnotationExtractor

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    txt = auto()
    json = auto()
    yaml = auto()


@dataclass
class Application(object):
    """Create and access discharge summary provenience of data annotations.

    """
    config_factory: ConfigFactory = field()
    """Used for prototyping."""

    log_conf: LogConfigurator = field()
    """The application log configuration utility."""

    corpus: Corpus = field()
    """The MIMIC-III corpus access object."""

    anon_resource: AnnotationResource = field()
    """Contains resources to acces the MIMIC-III MedSecId annotations."""

    reader: ProvenanceReader = field()
    """A utility class for accessing discharge summary *provenience of data*
    annotations.

    """
    writer: ProvenanceWriter = field()
    """Writes template files used by the physicians to annotate."""

    dsanon_stash: DischargeSummaryAnnotationStash = field()
    """A stash of :class:`.DischargeSummaryAnnotation` instances."""

    admission_similarity_assessor: AdmissionSimilarityAssessor = field()
    """Summarizes statistics and make sense of match data."""

    extractor: AnnotationExtractor = field()
    """Extracts the human annotations to a JSON file."""

    temporary_dir: Path = field()
    """A directory to put immediate temporary results."""

    def write_templates(self, limit: int = 1):
        """Write the files to be annotated by the physicians.

        :param limit: the max number of records to write

        """
        limit = sys.maxsize if limit is None else limit
        if 0:
            self.config_factory.config.write()
            return
        self.writer(limit)

    def write_template(self, hadm_id: str, output_path: Path = None):
        """Write an admission template by ID.

        :param hadm_id: the hospital admission ID

        :param output_path: where to save the file(s)

        """
        if output_path is None:
            output_path = Path('.') / hadm_id
        adm: HospitalAdmission = self.corpus.hospital_adm_stash[hadm_id]
        self.writer.write_admission(adm, output_path)

    def write_annotation(self, hadm_id: str):
        """Write an admission's annotations.

        :param hadm_id: the hospital admission ID

        :param output_path: where to save the file(s)

        """
        stash: Stash = self.admission_similarity_assessor.match_ann_stash
        mas: MatchedAnnotationSet = stash[hadm_id]
        mas.write()

    def extract_annotations(self, out_format: OutputFormat = OutputFormat.json,
                            include_text: bool = False,
                            include_sections: bool = False,
                            include_annotator: bool = False):
        """Extract annotations.

        :param out_format: the annotation output format

        :param include_text: whether to add the annotation's text

        :param include_sections: whether to add matched sections

        :param include_annotator whether to add the annotator identifier

        """
        out_file: Path = Path(f'annotations.{out_format.name}')
        meth: str = {
            OutputFormat.txt: 'write',
            OutputFormat.json: 'asjson',
            OutputFormat.yaml: 'asyaml',
        }[out_format]
        fn: Callable = getattr(self.extractor, meth)
        self.extractor.include_annotator = include_annotator
        self.extractor.include_text = include_text
        self.extractor.include_sections = include_sections
        with open(out_file, 'w') as f:
            fn(writer=f)
        logger.info(f'wrote: {out_file}')

    def keys(self):
        """Print the annotation keys of annotated admissions, which can then be
        used by ``read``.

        """
        k: str
        for k in self.dsanon_stash.keys():
            print(k)

    def read(self, hadm_id: str):
        """Read the discharge summary Word files annotated by the physicians.

        :param hadm_id: the hospital admission ID

        """
        dsanon: DischargeSummaryAnnotation = self.dsanon_stash[hadm_id]
        dsanon.write()

    def find_text_in_notes(self, text: str, hadm_id: str):
        """Find ``text`` by iterating through all notes of the admission.

        :param text: the text to find in the notes

        :param hadm_id: the hospital admission ID

        """
        self.admission_similarity_assessor.find_text_in_notes(text, hadm_id)

    def save_excel(self, output_path: Path = None):
        """Save all statistics data to an Excel file.

        :param output_path: where to save the file(s)

        """
        self.admission_similarity_assessor.save_excel(output_path)

    def save_csv(self, output_path: Path = None, yaml_dir: Path = None):
        """Save all statistics data as CSV files to a directory.

        :param output_path: where to save the file(s)

        :param yaml_dir: the directory where to save the yaml table config files

        """
        self.admission_similarity_assessor.save_csv(output_path, yaml_dir)

    def write_issues(self, stdout: bool = False):
        """Write all issues found in the discharge notes annotated corpus.

        :param stdout: whether to write the file to standard out

        """
        asa: AdmissionSimilarityAssessor = self.admission_similarity_assessor
        if stdout:
            # do not conflate real issues with recoverable warnings by turning
            # down the logger
            self.log_conf.level = 'err'
            self.log_conf()
            asa.write_issues(excludes={CommentIssue})
        else:
            adm_to_ann: Dict[str, str] = self.reader.annotators
            issues: Tuple[Issue] = asa.get_issues(excludes={CommentIssue})
            out_dir: Path = self.temporary_dir / 'issues'
            out_dir.mkdir(parents=True, exist_ok=True)
            for issue in issues:
                ann: str = adm_to_ann[issue.hadm_id]
                issue_file: Path = out_dir / f'{issue.hadm_id}-{ann}.txt'
                with open(issue_file, 'w') as f:
                    issue.write(writer=f)
                logger.info(f'wrote: {issue_file}')

    def _get_coverage_by_adm(self) -> DataFrameDescriber:
        dd: DataDescriber = self.admission_similarity_assessor.\
            get_data_describer()
        dd_name: str = 'Coverage by Admission'
        return next(iter(
            filter(lambda d: d.name == dd_name, dd.describers)))

    def _write_annotated_secid_predictions(self, cat_name: str = 'General'):
        """Write the original and predicted section IDs for a category for all
        annotated discharge summaries.

        :param cat_name: the category of the note to predict.

        """
        from typing import Tuple, List, Set
        from zensols.mimic import (
            Note, HospitalAdmission, HospitalAdmissionDbStash
        )
        from zensols.mimicsid import (
            PredictedNote, ApplicationFactory, AnnotationResource
        )
        from zensols.mimicsid.pred import SectionPredictor

        def write_note(org: Note, pred: Note, out_dir: Path):
            org_path: Path = out_dir / f'{org.row_id}-original.txt'
            pred_path: Path = out_dir / f'{org.row_id}-predicted.txt'
            with open(org_path, 'w') as f:
                f.write('original\n')
                org.write_fields(writer=f)
                f.write('_' * 60)
                f.write('\n')
                f.write(org.text)
            with open(pred_path, 'w') as f:
                f.write('prediction\n')
                org.write_fields(writer=f)
                pred.write(writer=f)

        out_dir: Path = self.temporary_dir / 'medsecid'
        out_dir.mkdir(parents=True, exist_ok=True)
        predictor: SectionPredictor = ApplicationFactory.section_predictor()
        ann_res: AnnotationResource = ApplicationFactory.annotation_resource()
        anon_row_ids: Set[str] = set(ann_res.note_ids['row_id'])
        astash: HospitalAdmissionDbStash = self.corpus.hospital_adm_stash
        dfd: DataFrameDescriber = self._get_coverage_by_adm()
        notes: List[Note] = []
        hadm_id: str
        for hadm_id in dfd.df['hadm_id']:
            adm: HospitalAdmission = astash[hadm_id]
            targ_notes: Tuple[Note] = adm.notes_by_category.get(cat_name)
            if targ_notes is not None:
                notes.extend(targ_notes)
        note_texts: Tuple[str] = tuple(map(lambda n: n.text, notes))
        preds = predictor(note_texts)
        pred: PredictedNote
        for org, pred in zip(notes, preds):
            if org.row_id in anon_row_ids:
                print(f'already annotated: {org}--skipping')
            else:
                write_note(org, pred, out_dir)
        logger.info(f'wrote: {len(note_texts)} {cat_name} notes')

    def write_annotator_sheet(self):
        """Write a spreadsheet with who is annotating what using note counts by
        admission.

        """
        annotators: List[str] = 'Kunal Aaron Sean'.split()
        df: pd.DataFrame = self.anon_resource.note_counts_by_admission
        df['annotator'] = tuple(it.islice(it.cycle(annotators), len(df)))
        df.to_excel(self.temporary_dir / 'admissions.xlsx', index=False)

    def write_medsecid_annotation_totals(self):
        """Write the MedSecId ID annotation and total counts to a spreadsheet.

        """
        from typing import Dict, Tuple
        from zensols.cli import CliHarness
        from zensols.mimic import Note, ApplicationFactory
        harn: CliHarness = ApplicationFactory.create_harness()
        non_ann_corpus: Corpus = harn['mimic_corpus']
        df_adm: pd.DataFrame = self.anon_resource.note_counts_by_admission
        #df_adm = df_adm.head(10)
        df_ont: pd.DataFrame = self.anon_resource.ontology
        cats: List[str] = df_ont['note_name'].drop_duplicates().to_list()
        cols: List[str] = list(map(self.anon_resource.category_to_id, cats))
        cols.insert(0, 'hadm_id')
        cols.append('total')
        rows: List[Tuple[str, ...]] = []
        hadm_id: str
        for hadm_id in df_adm['hadm_id']:
            adm: HospitalAdmission = non_ann_corpus.hospital_adm_stash[hadm_id]
            by_cat: Dict[str, Tuple[Note]] = adm.notes_by_category
            cnts = tuple(map(lambda c: len(by_cat[c]) if c in by_cat else 0,
                             cats))
            rows.append((hadm_id, *cnts, sum(cnts)))
        df = pd.DataFrame(rows, columns=cols)
        df = df_adm.merge(df, on='hadm_id', suffixes=('_ann', '_tot'))
        df.to_excel(self.temporary_dir / 'admissions-with-totals.xlsx',
                    index=False)

    def write_annotator_counts(self):
        """Write each annotator's discharge summary count."""
        dd = self.admission_similarity_assessor.get_data_describer()
        cnts: Dict[str, int] = dd['Coverage by Admission'].\
            df.groupby('annotator').size().to_dict()
        for ann, cnt in cnts.items():
            print(f'{ann}: {cnt}')
        print(f'total: {sum(cnts.values())}')

    def sentence_coverage(self, hadm_id: str):
        """Get the matched sentence coverage for an admission.

        :param hadm_id: the admission unique identifier, which must be annotated

        """
        from zensols.nlp import FeatureSentence
        from . import MatchedAnnotation, MatchedNote
        output_file: Path = self.temporary_dir / 'sentence-covergage.csv'
        stash: Stash = self.admission_similarity_assessor.match_ann_stash
        rows: List = []
        mas: MatchedAnnotationSet = stash[hadm_id]
        ma: MatchedAnnotation
        for ma in mas:
            mn: MatchedNote
            for mn in ma:
                s: FeatureSentence
                for s in mn.doc.get_overlapping_sentences(mn.span):
                    stoks = tuple(filter(lambda t: not t.is_space, s.token_iter()))
                    ssent = FeatureSentence(stoks)
                    otoks = s.get_overlapping_tokens(mn.span.narrow(s.lexspan))
                    otoks = tuple(filter(lambda t: not t.is_space, otoks))
                    osent = FeatureSentence(otoks)
                    rows.append((osent.token_len, ssent.token_len,
                                 ssent.canonical, osent.canonical))
        df = pd.DataFrame(rows, columns='n_overlap n_sent overlap sent'.split())
        df['portion'] = df['n_overlap'] / df['n_sent']
        df = df[df['n_overlap'] > 0]
        print(df.describe())
        df.to_csv(output_file)
        logger.info(f'wrote: {output_file}')

    def clear(self, all: bool = False):
        """Clear all cache files forcing rework on next invocation.

        :param all: whether to clear all cached data

        """
        self.admission_similarity_assessor.clear()
        if all:
            self.dsanon_stash.clear()
            self.writer.clear()

    def _tmp(self):
        from zensols.mimic import Note
        from . import MatchedNote
        stash = self.admission_similarity_assessor.match_ann_stash
        hadm_len = []
        for mas in stash.values():
            mn: MatchedNote = mas.matches[0].discharge_summary
            dn: Note = mn.note
            hadm_len.append((mn.hadm_id, len(dn.text)))
        df = pd.DataFrame(hadm_len, columns='hadm_id len'.split())
        print(df)

    def proto(self, run: int = 0):
        """For prototyping."""
        {0: self._tmp,
         1: lambda: self.write_issues(stdout=True),
         2: self.write_templates,
         3: self.save_excel,
         4: lambda: self.clear() or self.save_excel(),
         5: self.admission_similarity_assessor,
         6: lambda: self._sentence_coverage('139787'),
         }[run]()
