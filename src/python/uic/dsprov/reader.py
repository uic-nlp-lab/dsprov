"""Contains classes for reading discharge summary *provenience
of data* annotations.

"""
__author__ = 'Paul Landes'

from typing import List, Dict, Tuple, Set, Iterable, ClassVar
from dataclasses import dataclass, field
import logging
import sys
import re
from pathlib import Path
from io import StringIO, TextIOBase
import zipfile
from frozendict import frozendict
from lxml import etree
from lxml.etree import _Element as Element
from docx import Document
from zensols.persist import persisted, ReadOnlyStash
from zensols.config import Writable
from zensols.mimic import DischargeSummaryNote, Note, HospitalAdmission, Corpus
from . import (
    ProvenanceError, ProvenanceBase, Annotation, DischargeSummaryAnnotation
)

logger = logging.getLogger(__name__)


@dataclass
class _DischargeXmlReader(object):
    """Utillity classs to read the contents of the MS Word document with the
    discharge summary.  This reads the document by XML in the zipped ``docx``
    file.

    """
    _NS: ClassVar[str] = {
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    """XML namespace used in the Word doc file's XML files"""

    doc_path: Path = field()
    """The path to the MS Word document annotated by the physicians."""

    hadm_id: int = field()
    """The hospital admission ID of the discharge summary."""

    def get_elems(self, file_name: str, xpath: str) -> List[Element]:
        """Return a list of elements matching an XPath.

        :param file_name: the file name in the MS Word ``docx`` zip formatted
                          file

        :param xpath: the xpath matching elements to return

        """
        with zipfile.ZipFile(self.doc_path) as zipf:
            content = zipf.read(f'word/{file_name}.xml')
        et: Element = etree.XML(content)
        els: List[Element] = et.xpath(f'//w:{xpath}', namespaces=self._NS)
        return els

    def xpath(self, e: Element, xpath: str) -> List[Element]:
        """Return elements matching ``xpath`` starting at node ``e``."""
        return e.xpath(xpath, namespaces=self._NS)


@dataclass
class DischargeReader(ProvenanceBase, Writable):
    """Read the contents of the MS Word document with the discharge summary.

    """
    _DISCHARGE_FILE_REGEX: ClassVar[re.Pattern] = re.compile(
        r'^(\d+)-([a-z]{2})')
    _NEWLINE_LITERAL: ClassVar[str] = 'NEWLINE'

    hadm_id: str = field()
    """The hospital admission ID of the file to read."""

    @property
    @persisted('_word_file')
    def word_file(self) -> Path:
        """The path to the MS Word document file containing the discharge summary
        annotations.

        """
        paths: Tuple[Path] = tuple(
            filter(lambda d: d.name.startswith(self.hadm_id),
                   self.adm_dir.iterdir()))
        if len(paths) != 1:
            raise ProvenanceError(f'Expecting single path: {paths}')
        return paths[0]

    @property
    @persisted('_text')
    def text(self) -> str:
        """The full text of the discharge summary found in the MS Word
        document.

        """
        doc = Document(self.word_file)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return '\n'.join(text)

    @property
    @persisted('_discharge_reader')
    def discharge_reader(self) -> _DischargeXmlReader:
        """The discharge reader for this admission."""
        path: Path = self.word_file
        m: re.Pattern = self._DISCHARGE_FILE_REGEX.match(path.name)
        if m is None:
            raise ProvenanceError(
                f'Bad discharge summary file format: {path.name}')
        hadm_id, annotator = m.groups()
        return _DischargeXmlReader(path, hadm_id)

    def _parse_comments(self) -> Dict[str, Annotation]:
        """Return comments comment ID as keys and parsed comments values."""
        annotations: List[Annotation] = []
        dsr = self.discharge_reader
        ce: Element
        for ce in dsr.get_elems('comments', 'comment'):
            crs: int = 0
            ctext: str
            lines: List[str] = []
            te: Element
            for te in dsr.xpath(ce, 'w:p/w:r/w:t|w:p/w:r/w:cr'):
                tag: str = etree.QName(te).localname
                if tag == 't':
                    lines.append(te.text)
                elif tag == 'cr':
                    lines.append('\n')
                    crs += 1
            if crs > 0:
                ctext = ''.join(lines)
            else:
                ctext = '\n'.join(lines)
            ctext = ctext.replace('\n\n', '\n')
            ctext = ctext.replace(self._NEWLINE_LITERAL, '')
            annotations.append(Annotation(
                hadm_id=self.hadm_id,
                cid=dsr.xpath(ce, '@w:id')[0],
                author=dsr.xpath(ce, '@w:author')[0],
                date=dsr.xpath(ce, '@w:date')[0],
                comment=ctext))
        return frozendict({c.cid: c for c in annotations})

    @property
    @persisted('_annotations')
    def annotations(self) -> Tuple[Annotation]:
        """The annotated comments from the discharge summary document file.

        """
        dsr = self.discharge_reader
        text = StringIO()
        les: List[Element] = dsr.get_elems('document', 'p/*')
        comments: Dict[str, Annotation] = self._parse_comments()
        for le in les:
            tag: str = etree.QName(le).localname
            if tag == 'r':
                for te in dsr.xpath(le, '*'):
                    te_tag: str = etree.QName(te).localname
                    if te_tag == 't':
                        text.write(te.text)
                    elif te_tag == 'br':
                        text.write('\n')
            elif tag == 'commentRangeStart':
                cid: str = str(dsr.xpath(le, '@w:id')[0])
                com: Annotation = comments[cid]
                assert com.offset == -1
                com.offset = len(text.getvalue())
            elif tag == 'commentRangeEnd':
                cid: str = str(dsr.xpath(le, '@w:id')[0])
                com: Annotation = comments[cid]
                com_text: str = text.getvalue()
                com.text = com_text[com.offset:]
        for comment in comments.values():
            if comment.text is None:
                raise ProvenanceError(f'Missing comment text from <{comment}>')
        return tuple(comments.values())

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        for anon in self.annotations:
            self._write_line(f'comment: {anon.comment}', depth, writer)
            anon.write(depth + 1, writer)


@dataclass
class ProvenanceReader(ProvenanceBase):
    """A utility class for accessing discharge summary *provenience of data*
    annotations.

    """
    _FILE_REGEX: ClassVar[re.Pattern] = re.compile(r'^([0-9]+)-([a-z]{2})$')
    """The regular expression used to parse out the note ID"""

    @persisted('_parse_file_names_pw')
    def _parse_file_names(self) -> Tuple[Tuple[str, str], ...]:
        def parse_hadm(p: Path) -> Tuple[str, str]:
            m: re.Match = self._FILE_REGEX.match(p.stem)
            if m is None:
                logger.warning(f'bad file: {p}')
            else:
                return m.groups()

        return tuple(filter(lambda p: p is not None,
                            map(parse_hadm, self.adm_dir.iterdir())))

    @property
    @persisted('_hadm_ids')
    def hadm_ids(self) -> Set[str]:
        """The available hopsital admission IDs for reading."""
        return frozenset(map(lambda t: t[0], self._parse_file_names()))

    @property
    @persisted('_annotators')
    def annotators(self) -> Dict[str, str]:
        return frozendict(dict(self._parse_file_names()))

    def get_reader(self, hadm_id: str) -> DischargeReader:
        """Return a utility class that reads comments from the discharge notes
        annotations.

        :param hadm_id: the hospital admission ID of the file to read

        """
        return DischargeReader(self.adm_dir, hadm_id)


@dataclass
class DischargeSummaryAnnotationStash(ReadOnlyStash):
    """A stash of :class:`.DischargeSummaryAnnotation` instances parsed from the
    discharge summary annotated word file.

    """
    corpus: Corpus = field()
    """A container class for the resources that access the MIMIC-III corpus."""

    reader: ProvenanceReader = field()
    """Reads the discharge summary documents and their comments."""

    def load(self, hadm_id: str) -> DischargeSummaryAnnotation:
        if self.exists(hadm_id):
            adm_reader: DischargeReader = self.reader.get_reader(hadm_id)
            adm: HospitalAdmission = self.corpus.hospital_adm_stash[hadm_id]
            note: Note = adm.notes_by_category[DischargeSummaryNote.CATEGORY][0]
            assert adm_reader.text is not None
            anon = DischargeSummaryAnnotation(
                hadm_id=hadm_id,
                row_id=note.row_id,
                annotations=adm_reader.annotations,
                text=adm_reader.text,
                note_text=note.text,
                annotator=self.reader.annotators[hadm_id])
            return anon

    def keys(self) -> Iterable[str]:
        return self.reader.hadm_ids

    def exists(self, hadm_id: str) -> bool:
        return hadm_id in self.reader.hadm_ids
