"""Contains classes for creating discharge summary *provenience
of data* annotations.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass, field
import logging
from functools import reduce, cmp_to_key
import itertools as it
from pathlib import Path
import shutil
import pandas as pd
from docx import Document
from zensols.persist import FileTextUtil
from zensols.mimic import (
    RegexNote, DischargeSummaryNote, Note, HospitalAdmission, Corpus,
)
from zensols.mimicsid import (
    AnnotatedNote, MimicPredictedNote, AnnotationResource
)
from . import ProvenanceBase
from .reader import ProvenanceReader

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceWriter(ProvenanceBase):
    """Writes the *provenience of data* admission output files.

    """
    corpus: Corpus = field()
    """A container class for the resources that access the MIMIC-III corpus."""

    reader: ProvenanceReader = field()
    """A utility class for accessing discharge summary *provenience of data*
    annotations, and used to avoid writing admissions that have been completed.

    """
    anon_resource: AnnotationResource = field()
    """Contains resources to acces the MIMIC-III MedSecId annotations."""

    ascript: Path = field()
    """The path to the AppleScript that makes the MS Word document read-only."""

    adm_notes_file: Path = field()
    """The CSV of notes that look like admission notes per text search strings
    given by the physicians.

    """
    note_cat_per_admission_limit: int = field()
    """Limit for the number of unique types of note categories the admission must
    have to be an output candidate.

    """
    note_duplicate_length: int = field()
    """The first N characters used to compare notes, or the entire note text if
    ``None``.

    """
    def _get_as(self, doc_path: Path) -> str:
        with open(self.ascript) as f:
            return f.read().format(doc_path=str(doc_path.absolute()))

    def _filter_notes(self, adm: HospitalAdmission) -> List[Note]:
        def cmp_note(a: Note, b: Note) -> bool:
            an = isinstance(a, AnnotatedNote)
            bn = isinstance(b, AnnotatedNote)
            if an and not bn:
                return 1
            elif not an and bn:
                return -1
            elif a.storetime is not None and b.storetime is not None:
                return ((a.storetime > b.storetime) -
                        (a.storetime < b.storetime))
            else:
                return 0

        notes: Tuple[Note] = adm.notes
        nid: Dict[int, Note] = adm.notes_by_id
        dup_sets: Tuple[Set[str]] = adm.get_duplicate_notes(
            self.note_duplicate_length)

        if len(dup_sets) == 0:
            return notes
        else:
            dups: Set[str] = reduce(lambda x, y: x | y, dup_sets)

            # initialize with the notes not in any duplicate group, which are
            # non-duplicates
            non_dups: List[Note] = list(
                filter(lambda n: n.row_id not in dups, notes))
            ds: Set[str]
            for ds in dup_sets:
                dup_notes = sorted(
                    map(lambda r: nid[r], ds),
                    key=cmp_to_key(cmp_note),
                    reverse=True)
                non_dups.append(dup_notes[0])
            return non_dups

    def write_admission(self, adm: HospitalAdmission, output_dir: Path) -> Path:
        """Write the notes as text files and the discharge summary as an MS Word
        document for an admission.

        :param adm: the hospital admission

        :return: a path to the directory where the files were written

        """
        import applescript
        hadm_id: str = str(adm.hadm_id)
        fname: str = self._DISCHARGE_FILE_NAME.format(hadm_id=hadm_id)
        dis_path: Path = output_dir / fname
        ds: Note = adm.notes_by_category[DischargeSummaryNote.CATEGORY][0]
        doc = Document()
        doc.add_paragraph(ds.text)
        dis_path.parent.mkdir(parents=True)
        doc.save(dis_path)
        logger.info(f'wrote: {dis_path}')
        applescript.run(self._get_as(dis_path))
        for note in self._filter_notes(adm):
            cat: str = FileTextUtil.normalize_text(note.category)
            fname: str = f'{note.row_id}-{cat}.txt'
            note_path: Path = output_dir / 'raw' / fname
            fmt_path: Path = output_dir / 'formatted' / fname
            anonr: str = {
                AnnotatedNote: 'human',
                MimicPredictedNote: 'model',
            }.get(type(note))
            if anonr is None:
                if isinstance(note, RegexNote):
                    anonr = 'regular expression'
                else:
                    anonr = 'none'
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(f'Unknown note type {type(note)}: {note}')
            fmt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fmt_path, 'w') as f:
                note.write_fields(writer=f)
                f.write(f'annotator: {anonr}\n')
                note.write_sections(writer=f)
            logger.info(f'wrote {fmt_path}')
            note_path.parent.mkdir(parents=True, exist_ok=True)
            with open(note_path, 'w') as f:
                f.write(f'row_id: {note.row_id}\n')
                f.write(f'category: {note.category}\n')
                f.write(note.text)
            logger.info(f'wrote {note_path}')

    @property
    def adm_notes(self) -> pd.DataFrame:
        """A dataframe of all notes that look like admission notes per text search
        strings given by the physicians.

        """
        return pd.read_csv(self.adm_notes_file)

    def _get_latest(self, adm: HospitalAdmission,
                    category: str) -> Optional[Note]:
        notes: Tuple[Note] = adm.notes_by_category.get(category)
        if notes is not None and len(notes) > 0:
            notes = sorted(notes, key=lambda n: n.storetime, reverse=True)
            return notes[0]

    def clear(self):
        if self.adm_dir.is_dir():
            logger.info(f'removing: {self.adm_dir}')
            shutil.rmtree(self.adm_dir)

    def __call__(self, admission_limit: int):
        """Write all admissions matching the criteria given by
        :obj:`note_cat_per_admission_limit` from the annotated SecID
        annotations set.

        :param admission_limit: limit the written files to a number of
                                admissions

        """
        df: pd.DataFrame = self.anon_resource.note_counts_by_admission
        completed: Set[str] = self.reader.hadm_ids
        hadm_ids: Tuple[str] = tuple(
            it.islice(filter(lambda i: i not in completed, df['hadm_id']),
                      admission_limit))
        self.clear()
        logger.info(f'completed: {completed}--skipping these')
        hadm_id: str
        for hadm_id in it.islice(hadm_ids, admission_limit):
            adm: HospitalAdmission = self.corpus.hospital_adm_stash[hadm_id]
            adm_path: Path = self.adm_dir / hadm_id
            base_path: Path = adm_path / hadm_id
            self.write_admission(adm, base_path)
            logger.info(f'wrote {hadm_id} -> {adm_path}')
            zip_file: Path = Path(shutil.make_archive(
                hadm_id, 'zip', root_dir=adm_path))
            targ_zip_file: Path = self.adm_dir / zip_file.name
            zip_file.rename(targ_zip_file)
            shutil.rmtree(adm_path)
            logger.info(f'zipped: {targ_zip_file}')
