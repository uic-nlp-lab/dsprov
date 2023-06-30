"""Contains classes for creating and accessing discharge summary *provenience
of data* annotations.

The term *antecedent* for this module is a span of text found in the discharge
summary copied and pasted from the first originating note.

"""
__author__ = 'Paul Landes'

from typing import (
    List, Tuple, Set, Optional, ClassVar, Dict, Any, Iterable, Type
)
from dataclasses import dataclass, field
from abc import ABC, ABCMeta
import logging
from collections import OrderedDict
from pathlib import Path
import sys
from itertools import chain
import itertools as it
import textwrap
import re
from io import TextIOBase
from frozendict import frozendict
import pandas as pd
from zensols.util import APIError
from zensols.persist import persisted, PersistableContainer
from zensols.config import Dictable
from zensols.nlp import (
    LexicalSpan, FeatureSpan, TokenContainer, FeatureDocument
)
from zensols.mimic import (
    Note, Section, HospitalAdmission, DischargeSummaryNote, Corpus,
    NoteEvent, NoteEventPersister, NoteFactory
)

logger = logging.getLogger(__name__)


class ProvenanceError(APIError):
    """Thrown for all errors related to parsing and matching discharge notes
    text snippets to antecedent notes.

    """
    pass


class AdmissionError(ProvenanceError):
    """Raised for issues associated with an admission.

    """
    def __init__(self, hadm_id: str, msg: str):
        self.hadm_id = hadm_id
        super().__init__(msg)


class MissingNoteError(AdmissionError):
    """Raised for references to notes that don't belong from an admission.

    """
    def __init__(self, hadm_id: str, note_id: str, text: str):
        self.note_id = note_id
        self.text = text
        super().__init__(
            hadm_id=hadm_id,
            msg=(f'Missing note: {note_id}, admission: {hadm_id} ' +
                 f'for text: <{text}>'))


@dataclass
class Issue(Dictable):
    """Contains data about some issue that's wrong with annotated data.

    """
    hadm_id: str = field()
    """The admission ID of the issue notes."""

    desc: str = field()
    """The description of the issue."""

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'{self.hadm_id}: {self.desc}', depth, writer)


@dataclass
class IssueContainer(Dictable):
    """Contains data that potentially has issues that should be reported.

    """
    def get_issues(self, excludes: Set[Type[Issue]] = None) -> Tuple[Issue]:
        issues = []
        try:
            self._add_issues(issues)
        except AdmissionError as e:
            issues.append(Issue(e.hadm_id, str(e)))
        if excludes is not None:
            issues = filter(lambda i: type(i) not in excludes, issues)
        return tuple(issues)

    def _add_issues(self, issues: List[Issue]):
        """Add issues (errors) for this discharge note annotation.

        :param issues: to be populated with any issues

        """
        pass

    def write_issues(self, depth: int = 0, writer: TextIOBase = sys.stdout,
                     excludes: Set[Type[Issue]] = None):
        issues = self.get_issues(excludes)
        if len(issues) > 0:
            self._write_line('issues:', depth, writer)
            self._write_iterable(issues, depth + 1, writer, include_index=True)


@dataclass
class _MatchResource(object):
    corpus: Corpus = field()
    """A container class for the resources that access the MIMIC-III corpus."""

    mimic_note_factory: NoteFactory = field(repr=False)
    """The factory that creates :class:`.Note` for hopsital admissions."""

    def get_adm(self, hadm_id: str) -> HospitalAdmission:
        return self.corpus.hospital_adm_stash[hadm_id]


@dataclass(repr=False)
class AnnotationBase(PersistableContainer, IssueContainer, metaclass=ABCMeta):
    """A base class for all match data container classes.

    """
    _PERSITABLE_TRANSIENT_ATTRIBUTES: ClassVar[Set] = {'_resource'}

    def __post_init__(self):
        super().__init__()

    def _set_resource(self, resource: _MatchResource):
        self._resource = resource

    def get_source_note(self, row_id: str, note_regex: bool = True) -> Note:
        """Return a :class:`~zensols.mimic.note.Note` from the information of a
        :class:`~zensols.mimic.note..NoteAntecedent`.

        :param note_regex: whether to return an instance a
                           :class:`~zensols.mimic.note.RegexNote` (if possible
                           based on the category) or the ``Note`` base class

        """
        note: Note
        if note_regex:
            note = self._resource.corpus.get_note_by_id(row_id)
        else:
            corpus: Corpus = self._resource.corpus
            note_factory: NoteFactory = self._resource.mimic_note_factory
            persister: NoteEventPersister = corpus.note_event_persister
            note_event: NoteEvent = persister.get_by_id(row_id)
            note = note_factory.create(
                note_event,
                section=note_factory.mimic_default_note_section)
        return note

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(repr=False)
class Annotation(AnnotationBase):
    """Represents an annotation created by the physicans in the document.

    An annotation is either a comment about what's been selected in the
    document, or a specific format containing the originating note antecedent
    text. In this case, the comment text has the form::

        <note ID>:<copy/pasted>

    """
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = {'span', 'note_id', 'note_text'}

    _COMMENT_REGEX: ClassVar[re.Pattern] = re.compile(
        r'^(\d+)\s*:\s*(.+)$', re.DOTALL)
    """Parss note ID (row_id) and remainder of the copy/pasted text."""

    hadm_id: str = field()
    """The admission ID of the discharge notes."""

    cid: int = field()
    """The comment ID."""

    author: str = field(repr=False)
    """Who wrote the comment."""

    date: str = field(repr=False)
    """Date and time the comment was written."""

    comment: str = field(repr=False)
    """The text of the comment provided by the physician."""

    offset: str = field(default=-1)
    """The offset of the comment in the discharge summary."""

    text: str = field(default=None)
    """"The text selected in the discharge summary for the comment."""

    @property
    @persisted('_span', transient=True)
    def span(self) -> LexicalSpan:
        """The beginning and end offsets in the discharge summary of the
        selected text for the respective comments.

        """
        return LexicalSpan(self.offset, self.offset + len(self.text))

    @persisted('_parsed_comment', transient=True)
    def _get_parsed_comment(self) -> Optional[Tuple[int, str]]:
        m: re.Match = self._COMMENT_REGEX.match(self.comment)
        if m is not None:
            return int(m.group(1)), m.group(2)

    @property
    def note_id(self) -> Optional[int]:
        """The note ID parsed from the :obj:`comment` text, which is the
        ``row_id`` column in the MIMIC III database.

        :return: the note ID as a parsed integer if the comment was antecedent
                 formatted (see class docs)

        """
        parts: Optional[Tuple[int, str]] = self._get_parsed_comment()
        return None if parts is None else parts[0]

    @property
    def note_text(self) -> str:
        """The note text parsed from the :obj:`comment` text, which is the text
        found in the related (non-discharge) note.

        """
        parts: Optional[Tuple[int, str]] = self._get_parsed_comment()
        return self.comment if parts is None else parts[1]

    @property
    def is_comment(self) -> bool:
        """Whether this annotation is a notation comment rather than a text
        match.

        """
        return self.note_id is None

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        dct = self.asdict()
        for k in 'cid offset span text note_id note_text'.split():
            del dct[k]
        self._write_line(f'cid: {self.cid}', depth, writer)
        self._write_dict(dct, depth + 1, writer)
        if self.note_id is not None:
            self._write_line(f'note_id: {self.note_id}', depth + 1, writer)
        self._write_line(f'note_text: {self.note_text}', depth + 1, writer,
                         max_len=self.WRITABLE_MAX_COL)
        if self.text is not None:
            self._write_line('text:', depth + 1, writer)
            self._write_block(self.text, depth + 2, writer, limit=5)
        self._write_line(f'span: {self.span}', depth + 1, writer)

    def __str__(self) -> str:
        text = f'comment: {textwrap.shorten(self.note_text, 30)}'
        if self.text is not None:
            text += f', text: {textwrap.shorten(self.text, 40)}'
        return f'{self.cid}: note={self.note_id}, {text}'


@dataclass
class AnnotationIssue(Issue):
    """An issue that stems from an annotation.

    """
    annotation: Annotation = field()
    """The annotation with the issue."""

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write(depth, writer)
        self._write_line('annotation:', depth, writer)
        self._write_object(self.annotation, depth + 1, writer)


@dataclass
class CommentIssue(AnnotationIssue):
    """An annotation that is interpreted as a comment because it is not
    formatted as a match.

    """
    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write(depth, writer)
        self._write_line(f'comment: <{self.annotation.comment}>',
                         depth, writer)


@dataclass
class DischargeSummaryAnnotation(AnnotationBase):
    """Represents the annotations created by the physicans in a discharge
    summary document.

    """
    hadm_id: str = field()
    """The admission ID of the discharge notes."""

    row_id: int = field()
    """The discharge note DB ``note_id`` identifier."""

    annotations: Tuple[Annotation] = field()
    """The annotations provided by the physicians."""

    text: str = field(repr=False)
    """The full text of the discharge summary from the document."""

    note_text: str = field(repr=False)
    """The full text of the discharge summary from the database."""

    annotator: str = field()
    """The human annotator."""

    @property
    def comments(self) -> Tuple[Annotation]:
        """Return notation comments.

        :see: :obj:`text_match_annotations`

        """
        return tuple(filter(lambda n: n.is_comment, self.annotations))

    @property
    def text_match_annotations(self) -> Tuple[Annotation]:
        """Return text matches.

        :see :obj:`comments`

        """
        return tuple(filter(lambda n: not n.is_comment, self.annotations))

    def _add_issues(self, issues: List[Issue]):
        """Add issues (errors) for this discharge note annotation.

        :param issues: to be populated with any issues

        """
        for anon in self.annotations:
            ds_snip: str = self.text[anon.span[0]:anon.span[1]]
            if anon.text != ds_snip:
                s = 'Comment text and discharge summary does not match'
                issues.append(AnnotationIssue(self.hadm_id, s, anon))
            if anon.is_comment:
                s = 'Interpreting as a comment'
                issues.append(CommentIssue(self.hadm_id, s, anon))

        if self.text != self.note_text:
            s = 'Annotated and DB discharge summary note text do not match'
            issues.append(Issue(s))
        for anon in self.annotations:
            ds_snip: str = self.note_text[anon.span[0]:anon.span[1]]
            if anon.text != ds_snip:
                issues.append('Comment annotation text does not match in ' +
                              f'discharge summary note: {anon}')

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              limit: int = sys.maxsize):
        self._write_line(f'hadm_id: {self.hadm_id}', depth, writer)
        if limit > 0:
            self._write_line('annotations:', depth, writer)
            anons = it.islice(self.annotations, limit)
            self._write_iterable(anons, depth + 1, writer)
        super().write_issues(depth, writer)

    def __getitem__(self, i: int) -> Annotation:
        return self.comments[i]

    def __str__(self) -> str:
        text = textwrap.shorten(self.text, self.WRITABLE_MAX_COL)
        return f'{self.hadm_id}: {text}'


@dataclass
class NoteAntecedent(AnnotationBase):
    """A match from the discharge summary selected/commented document text
    linked to specific lexical spans in the antecedent note.  Each of the
    :obj:`spans` specifies lexical spans of text that matches the discharge note
    (:obj:`.Annotation.note_text`).

    """
    _DICTABLE_WRITABLE_DESCENDANTS: ClassVar[bool] = True
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = {'matched'}

    annotation: Annotation = field()
    """The discharge note annotation that refers to this antecedent."""

    spans: Tuple[LexicalSpan] = field()
    """The text spans having the same text found in the
    :obj:`.Annotation.note_text` (see class docs).

    """
    @property
    def matched(self) -> bool:
        """Whether this instance represents a match."""
        return len(self.spans) > 0


@dataclass
class MissIssue(Issue):
    """An issue generated for :class:`.NoteAntecedent`s with no matches.

    :see: :obj:`.NoteAntecedent.matched`

    """
    miss: NoteAntecedent = field()
    """The antecedent with no matches."""

    note: Note = field()
    """The note instance of the antecedent."""

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_object(self.miss, depth, writer)
        self._write_divider(depth, writer, char='-', width=10)
        self._write_line('note text from the comment:', depth, writer)
        self._write_block(self.miss.annotation.note_text, depth, writer)
        self._write_divider(depth, writer, char='_')
        self._write_line(f'note: {self.miss.annotation.note_id}', depth, writer)
        self._write_block(self.note.text, depth, writer)
        self._write_divider(depth, writer, char='=')


@dataclass
class NoteAntecedentSet(AnnotationBase):
    """Contains all the matches (and misses) from discharge annotator's
    copy/paste text snippets to antecedent medical notes.

    """
    _DICTABLE_WRITABLE_DESCENDANTS: ClassVar[bool] = True

    discharge_summary_annotation: DischargeSummaryAnnotation = field()
    """The discahrge annotation to where the text was copied."""

    antecedents: Tuple[NoteAntecedent] = field()
    """The antecedents for the match set."""

    def _set_resource(self, resource: _MatchResource):
        super()._set_resource(resource)
        self.discharge_summary_annotation._set_resource(resource)
        for ant in self.antecedents:
            ant._set_resource(resource)

    @property
    @persisted('_admission', transient=True)
    def admission(self) -> HospitalAdmission:
        """The admission for this antecedent set."""
        hadm_id: str = self.discharge_summary_annotation.hadm_id
        return self._resource.get_adm(hadm_id)

    @property
    def discharge_summary(self) -> Note:
        """The source discharge summary note instance for this set."""
        return self.admission[self.discharge_summary_annotation.row_id]

    def write_antecedents(self, depth: int = 0,
                          writer: TextIOBase = sys.stdout):
        match: NoteAntecedent
        for i, match in enumerate(self.antecedents):
            if i > 0:
                self._write_divider(depth, writer, char='_')
            self._write_object(match, depth, writer)

    def _add_issues(self, issues: List[Issue]):
        hadm_id: str = self.discharge_summary_annotation.hadm_id
        adm: HospitalAdmission = self._resource.get_adm(hadm_id)
        antecedents: List[NoteAntecedent] = self.antecedents
        misses = enumerate(filter(lambda m: not m.matched, antecedents))
        self.discharge_summary_annotation._add_issues(issues)
        miss: NoteAntecedent
        for i, miss in misses:
            issue = MissIssue(
                hadm_id=hadm_id,
                desc='Discharge summary not matched to note antecedent',
                miss=miss,
                note=adm[miss.annotation.note_id])
            issues.append(issue)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line('antecedents:', depth, writer)
        self.write_antecedents(depth, writer)
        self._write_divider(depth, writer, char='_')
        self.write_issues(depth, writer)

    def __str__(self):
        return (f'{self.discharge_summary_annotation}, ' +
                f'{len(self.antecedents)} antecedents')


@dataclass(repr=False)
class SpanCounts(AnnotationBase):
    """A container class with the counts of tokens and characters for a span of
    text.  This span can include an entire document or note.

    """
    _DICTABLE_WRITABLE_DESCENDANTS: ClassVar[bool] = True

    tokens: int = field()
    """The number of tokens in the span."""

    chars: int = field()
    """The number of characters in the span."""

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(str(self), depth, writer)

    def __iter__(self) -> Iterable[int]:
        return iter((self.tokens, self.chars))

    def __str__(self) -> str:
        return f'tokens: {self.tokens}, chars: {self.chars}'


@dataclass(repr=False)
class SectionSpanCounts(AnnotationBase):
    """A sections span counts.

    """
    id: str = field()
    """The section ID."""

    header: SpanCounts = field()
    """The section header counts."""

    body: SpanCounts = field()
    """The section body counts."""

    @property
    def counts(self) -> SpanCounts:
        return SpanCounts(self.header.tokens + self.body.tokens,
                          self.header.chars + self.body.chars)

    def __iter__(self) -> Iterable[int]:
        return self.counts.__iter__()

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_id: bool = True):
        if include_id:
            self._write_line(self.id, depth, writer)
            depth += 1
        self._write_line(f'header: {self.header}', depth, writer)
        self._write_line(f'body: {self.body}', depth, writer)


@dataclass(repr=False)
class MatchedSection(AnnotationBase):
    """The text that matches between the discharge summary and the antecedent
    note.  The token data only includes that proper subest of tokens that
    overlap the span given by annotation span(s), These annotation span(s) are
    :obj:`.Annotation.span` for discharge summaries or
    :obj:`.NoteAntecedent.spans` for note antecedents.

    """
    _PERSITABLE_TRANSIENT_ATTRIBUTES: ClassVar[Set] = \
        AnnotationBase._PERSITABLE_TRANSIENT_ATTRIBUTES | {'_matched_note'}

    section_id: str = field()
    """The matched section."""

    text: str = field()
    """The annotation text.  This is the :obj:`.Annotation.note_text` for
    discharge summaries and :obj:`.Annotation.text` for note antecedents.

    """
    header_spans: Tuple[LexicalSpan] = field()
    """The narrowed span across the section header and matched span."""

    header_tokens: FeatureSpan = field()
    """The proper subset of section tokens that matched the header of the
    section.

    """
    body_span: LexicalSpan = field()
    """The narrowed span across the section body and matched span."""

    matched_text: str = field(default=None)
    """A string of the contiguous spans of text from :obj:`body_span` and
    :obj:`header_spans`.

    """
    def __post_init__(self):
        super().__post_init__()
        self._matched_note = None

    @property
    @persisted('_tokens')
    def tokens(self) -> TokenContainer:
        spans: List[LexicalSpan] = list(self.header_spans)
        if len(self.body_span) > 0:
            spans.append(self.body_span)
        span: LexicalSpan = LexicalSpan.widen(spans)
        if self._matched_note is None:
            return FeatureDocument.EMPTY_DOCUMENT
        return self._matched_note.doc.get_overlapping_document(span)

    @property
    @persisted('_body_tokens')
    def body_tokens(self) -> TokenContainer:
        if self._matched_note is None or len(self.body_span) == 0:
            return FeatureDocument.EMPTY_DOCUMENT
        return self._matched_note.doc.get_overlapping_document(self.body_span)

    @property
    def counts(self) -> SectionSpanCounts:
        hs = self.header_spans
        hl: int = 0 if hs is None else sum(map(len, hs))
        bl: int = 0 if self.body_span is None else len(self.body_span)
        return SectionSpanCounts(
            id=self.section_id,
            header=SpanCounts(self.header_tokens.token_len, hl),
            body=SpanCounts(self.body_tokens.token_len, bl))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        header: str = ' '.join(map(lambda t: t.text, self.header_tokens))
        body: str = ' '.join(map(lambda t: t.text, self.body_tokens))
        self._write_line(f'section: {self.section_id}', depth, writer)
        self._write_line(f'text: <{self.text}>', depth, writer, max_len=True)
        self._write_line(f'header: <{header}>', depth, writer, max_len=True)
        self._write_line(f'body: <{body}>', depth, writer, max_len=True)
        self._write_line('counts:', depth, writer)
        self.counts.write(depth + 1, writer, include_id=False)

    def _from_dictable(self, *args, **kwargs) -> Dict[str, Any]:
        return self.counts.asdict()

    def __str__(self) -> str:
        return str(self.counts)


@dataclass(repr=False)
class MatchedNote(AnnotationBase):
    """The matched content in several forms: raw text, tokens and sections.
    Where this content comes from is either the discharge summary text or the
    source antecedent note text.  For discharge summaries, this is the selected
    text.  For the antecedent note text, this is the comment text in the
    annotated discharge summary documents that matches the antecedent note text.

    """
    hadm_id: str = field()
    """The admission ID of the discharge notes."""

    row_id: int = field()
    """The unique note ID from the ``noteevents`` table."""

    cid: int = field()
    """The comment ID."""

    text: str = field()
    """The annotation text, which useful when reporting issues or debugging.
    This is the :obj:`.Annotation.note_text` (comment) for discharge summaries
    and :obj:`.Annotation.text` (text selected in the discharge summary) for
    note antecedents.

    """
    note_category: str = field()
    """The category of the note."""

    span: LexicalSpan = field()
    """The span of the note's match."""

    sections: Tuple[MatchedSection] = field()
    """The matched section data.  Discharge summary instances always have
    exactly one.

    """
    def __post_init__(self):
        super().__post_init__()
        self._reset_sections()

    def _reset_sections(self):
        sec: MatchedSection
        for sec in self.sections:
            sec._matched_note = self

    def _set_resource(self, resource: _MatchResource):
        super()._set_resource(resource)
        sec: MatchedSection
        for sec in self.sections:
            sec._set_resource(resource)

    @property
    def admission(self) -> HospitalAdmission:
        """The hospital admission of this matched note."""
        return self._resource.get_adm(self.hadm_id)

    @property
    def note(self) -> Note:
        """The note instance of the match."""
        return self.admission[self.row_id]

    @property
    def doc(self) -> FeatureDocument:
        """The feature document of the note."""
        return self.note.doc

    @property
    def note_text(self) -> str:
        """The entire text of the note."""
        return self.note.text

    @property
    @persisted('_tokens', transient=True)
    def tokens(self) -> TokenContainer:
        """The tokens that overlap the span to the matched text (see class
        docs).

        """
        return self.doc.get_overlapping_document(self.span, False)

    @property
    def counts(self) -> SpanCounts:
        """The span counts for the match."""
        toks: TokenContainer = self.tokens
        return SpanCounts(toks.token_len, len(toks.text))

    @property
    def span_text(self) -> str:
        """The :obj:`note_text` substring demarcated by :obj:`span`."""
        s = self.span
        return self.note_text[s.begin:s.end]

    def _add_issues(self, issues: List[Issue]):
        if self.note_category == DischargeSummaryNote.CATEGORY:
            d = ('Discharge summary used as the note antecedent ' +
                 f'for note {self.row_id}')
            issues.append(Issue(self.hadm_id, d))
        if len(self.sections) == 0:
            txt: str = textwrap.shorten(self.span_text, 40)
            d = (f'No sections with text, adm: {self.hadm_id}, ' +
                 f'<{txt}>: note {self.row_id}')
            issues.append(Issue(self.hadm_id, d))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'row_id: {self.row_id}', depth, writer)
        self._write_line(f'text: {self.text}', depth + 1, writer,
                         max_len=True, repl_newlines=True)
        self._write_line(f'tokens: <{self.tokens.text}>',
                         depth + 1, writer, max_len=True)
        self._write_line('counts:', depth, writer)
        self._write_object(self.counts, depth + 2, writer)
        self._write_line('sections:', depth, writer)
        self._write_iterable(self.sections, depth + 2, writer)

    def _from_dictable(self, *args, **kwargs) -> Dict[str, Any]:
        return OrderedDict(
            [['row_id', self.row_id],
             ['counts', self.counts.asdict()],
             ['sections', tuple(map(lambda s: s.asdict(), self.sections))],
             ['text', OrderedDict(
                 [['token_text', ' '.join(map(lambda t: t.text, self.tokens))],
                  ['text', self.text]])]])

    def __setstate__(self, state: Dict[str, Any]):
        super().__setstate__(state)
        self._reset_sections()

    def __str__(self):
        text = textwrap.shorten(self.text, 30)
        note_text = textwrap.shorten(self.note_text, 30)
        return (f'hadm_id: {self.hadm_id}, row_id: {self.row_id}, ' +
                f'cid: {self.cid}, category: {self.note_category}, ' +
                f'sections: {len(self.sections)}, ' +
                f'text: {text}, note text: {note_text}')


@dataclass(repr=False)
class MatchedAnnotation(AnnotationBase):
    """A matched text snipped from the discharge summary to the antecendent
    note.

    """
    discharge_summary: MatchedNote = field()
    """The match data from the discharge summary."""

    note_antecedents: Tuple[MatchedNote] = field()
    """The match data from the note antecedent."""

    annotator: str = field()
    """The human annotator."""

    def _set_resource(self, resource: _MatchResource):
        super()._set_resource(resource)
        self.discharge_summary._set_resource(resource)
        sec: MatchedSection
        for ant in self.note_antecedents:
            ant._set_resource(resource)

    def _add_issues(self, issues: List[Issue]):
        ant_match: MatchedNote
        for ant_match in self.note_antecedents:
            ant_match._add_issues(issues)

    def write_text(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        """A simpliifed format that includes only the text of the match."""
        self._write_line('discharge summary:', depth, writer)
        self._write_block(self.discharge_summary.text, depth + 1, writer)
        self._write_line('antecedents:', depth, writer)
        mn: MatchedNote
        for i, mn in enumerate(self.note_antecedents):
            self._write_line(f'{mn.row_id}:', depth + 1, writer)
            if i > 0:
                self._write_divider(depth + 1, writer)
            self._write_block(mn.text, depth + 2, writer)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'annotator: {self.annotator}', depth, writer)
        self._write_line('discharge summary:', depth, writer)
        self._write_object(self.discharge_summary, depth + 1, writer)
        self._write_line('note antecedents:', depth, writer)
        self._write_iterable(self.note_antecedents, depth + 1, writer,
                             include_index=True)

    def __len__(self):
        return len(self.note_antecedents) + 1

    def __iter__(self) -> Iterable[MatchedNote]:
        return chain.from_iterable(
            [[self.discharge_summary], self.note_antecedents])

    def __str__(self) -> str:
        return (f'{self._trunc(str(self.discharge_summary))}, ' +
                f'({len(self.note_antecedents)}) antecedents')


@dataclass(repr=False)
class MatchedAnnotationSet(AnnotationBase):
    """The matches between the discharge summaries and note antecedents that
    represent the entire set for one discharge summary annotated document.

    """
    hadm_id: str = field()
    """The admission ID of the discharge notes."""

    matches: Tuple[MatchedAnnotation] = field()
    """The matched annotations betwee the discharge summary the note
    antecedents.

    """
    dataframe: pd.DataFrame = field()
    """Contains this instance's data in tabular format."""

    def _set_resource(self, resource: _MatchResource):
        super()._set_resource(resource)
        ma: MatchedAnnotation
        for ma in self.matches:
            ma._set_resource(resource)

    @staticmethod
    def _counts(row_id: int, adm: HospitalAdmission) -> SpanCounts:
        note: Note = adm[row_id]
        doc: FeatureDocument = note.doc
        return SpanCounts(doc.token_len, len(doc.text))

    def get_note(self, matched_note: MatchedNote) -> Note:
        """Get the MIMIC note that belongs to a match.

        :param matched_note: used to find the MIMIC note using
                            :obj:`matched_note.hadm_id` and
                            :obj:`matched_note.row_id`

        :return: the note instance for matched_note

        """
        hadm_id: str = matched_note.hadm_id
        adm: HospitalAdmission = self._resource.get_adm(hadm_id)
        return adm[matched_note.row_id]

    @property
    def admission(self) -> HospitalAdmission:
        """The admission for which matches apply."""
        return self._resource.get_adm(self.hadm_id)

    @property
    def discharge_summary_row_id(self) -> int:
        """The ``hadm_id`` of the discharge summary for this matching."""
        if len(self.matches) == 0:
            raise ProvenanceError(f'No matches found for {self}')
        return self.matches[0].discharge_summary.row_id

    @property
    def discharge_summary_note(self) -> Note:
        """The matched discharge summary note."""
        return self.admission[self.discharge_summary_row_id]

    @property
    def discharge_summary_counts(self) -> SpanCounts:
        """The discharge summary counts of the entire note."""
        return self._counts(self.discharge_summary_row_id, self.admission)

    @property
    def discharge_summary_section_counts(self) -> Dict[str, SpanCounts]:
        """The discharge summary's total counts for each section keyed by
        section ID.

        """
        note: Note = self.admission[self.discharge_summary_row_id]
        scounts: List[SectionSpanCounts] = []
        sec: Section
        for sec in note.sections.values():
            htoks = 0
            hcounts = 0
            for hspan in sec.header_spans:
                htoks += len(note.doc.get_overlapping_tokens(hspan))
                hcounts += len(hspan)
            hcounts = SpanCounts(htoks, hcounts)
            btoks = tuple(note.doc.get_overlapping_tokens(sec.body_span))
            bcounts = SpanCounts(len(btoks), len(sec.body_span))
            scounts.append(SectionSpanCounts(sec.id, hcounts, bcounts))
        return frozendict({x.id: x for x in scounts})

    @property
    def note_antecedent_counts(self) -> Dict[int, SpanCounts]:
        """The counts all matched note antecedent keyed by ``row_id``."""
        adm: HospitalAdmission = self.admission
        ant_row_ids: Set[int] = set()
        ant_counts: Dict[str, SpanCounts] = {}
        match: MatchedAnnotation
        for match in self.matches:
            ant_row_ids.update(map(lambda a: a.row_id, match.note_antecedents))
        row_id: int
        for row_id in ant_row_ids:
            ant_counts[row_id] = self._counts(row_id, adm)
        return frozendict(ant_counts)

    def _add_issues(self, issues: List[Issue]):
        match: MatchedAnnotation
        for match in self.matches:
            match._add_issues(issues)

    def write_text(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        """A simpliifed format that includes only the text of the match."""
        self._write_line(f'hadm_id: {self.hadm_id}', depth, writer)
        self._write_line('annotations:', depth, writer)
        ma: MatchedAnnotation
        for i, ma in enumerate(self):
            self._write_divider(depth + 1, writer, header=f'<{i}>')
            ma.write_text(depth + 1, writer=writer)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._keep_sections_object = True
        try:
            dct = self.asdict()
        finally:
            del self._keep_sections_object
        ds = dct['discharge_summary']
        secs: Dict[str, SectionSpanCounts] = ds.pop('sections')
        self._write_line('discharge summary:', depth, writer)
        self._write_dict(ds, depth + 1, writer)
        sec: SectionSpanCounts
        for sid, sec in secs.items():
            self._write_line(sid, depth + 1, writer)
            sec.write(depth + 2, writer, include_id=False)
        self._write_line('note antecedents:', depth, writer)
        self._write_dict(dct['note_antecedents'], depth + 1, writer)
        self._write_line('matches:', depth, writer)
        self._write_iterable(self.matches, depth + 1, writer,
                             include_index=True)

    def _from_dictable(self, *args, **kwargs) -> Dict[str, Any]:
        def map_dict(d: Dict) -> Dict:
            del d['id']
            return d

        if not hasattr(self, '_keep_sections_object'):
            sections = dict(map(lambda c: (c[0], map_dict(c[1].asdict())),
                                self.discharge_summary_section_counts.items()))
        else:
            sections = self.discharge_summary_section_counts
        return OrderedDict(
            [['discharge_summary', OrderedDict(
                [['hadm_id', self.hadm_id],
                 ['row_id', self.discharge_summary_row_id],
                 ['counts', self.discharge_summary_counts.asdict()],
                 ['sections', sections]])],
             ['note_antecedents',
              dict(map(lambda s: (s[0], s[1].asdict()),
                       self.note_antecedent_counts.items()))],
             ['matches', tuple(map(lambda m: m.asdict(), self.matches))]])

    def __getitem__(self, i: int) -> MatchedAnnotation:
        return self.matches[i]

    def __iter__(self) -> Iterable[MatchedAnnotation]:
        return iter(self.matches)


@dataclass
class ProvenanceBase(ABC):
    """A base class for *provenience of data* classes that have a directory that
    has the admission output files.

    """
    _DISCHARGE_FILE_NAME: ClassVar[str] = '{hadm_id}-discharge.docx'
    """The path to the annotated discharge note document."""

    adm_dir: Path = field()
    """The path where the annotation discharge summary and reference notes are
    written.

    """
