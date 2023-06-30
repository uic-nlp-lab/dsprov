"""Automatic methods to match notes.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Iterable, List, ClassVar, Optional
from dataclasses import dataclass, field
import sys
import logging
from io import TextIOBase
from zensols.persist import Stash
from zensols.config import Dictable
from zensols.nlp import (
    FeatureSpan, FeatureSentence, FeatureDocument,
    DecoratedFeatureDocumentParser
)
from zensols.spanmatch import Match, MatchResult, Matcher
from . import (
    AdmissionError, MatchedNote, MatchedAnnotation, MatchedAnnotationSet,
    AdmissionSimilarityAssessor,
)

logger = logging.getLogger(__name__)


@dataclass
class NoteMatch(Dictable):
    _SRC_TARG_DOC: ClassVar[List[str]] = 'discharge_summary antecedent'.split()

    hadm_id: str = field()
    comment_id: int = field()
    discharge_summary_doc: FeatureDocument = field()
    antecedent_doc: FeatureDocument = field()
    discharge_summary_span_gold: FeatureSpan = field()
    antecedent_span_gold: FeatureSpan = field()

    @classmethod
    def reverse(cls: NoteMatch):
        cls._SRC_TARG_DOC.reverse()

    @classmethod
    def source_target_names(cls: NoteMatch) -> List[str]:
        return cls._SRC_TARG_DOC

    @property
    def source_doc(self) -> FeatureDocument:
        return getattr(self, f'{self._SRC_TARG_DOC[0]}_doc')

    @property
    def target_doc(self) -> FeatureDocument:
        return getattr(self, f'{self._SRC_TARG_DOC[1]}_doc')

    @property
    def source_span(self) -> FeatureSpan:
        return getattr(self, f'{self._SRC_TARG_DOC[0]}_span_gold')

    @property
    def target_span(self) -> FeatureSpan:
        return getattr(self, f'{self._SRC_TARG_DOC[1]}_span_gold')

    @property
    def key(self) -> str:
        return f'{self.hadm_id}.{self.comment_id}'

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(f'{self.hadm_id}:', depth, writer)
        self._write_line('gold:', depth + 1, writer)
        self._write_line('discharge:', depth + 2, writer)
        if self.discharge_summary_span_gold is not None:
            self.discharge_summary_span_gold.write(
                depth + 3, writer, n_tokens=0)
        self._write_line('antecedent:', depth + 2, writer)
        if self.antecedent_span_gold is not None:
            self.antecedent_span_gold.write(depth + 3, writer, n_tokens=0)
        self._write_line('auto:', depth + 1, writer)


@dataclass
class PredictionMatcher(object):
    doc_parser: DecoratedFeatureDocumentParser = field()
    """The document parser that decorates documents needed to be processed by
    :obj:`matcher`.

    """
    matcher: Matcher = field()
    """The document matcher."""

    admission_similarity_assessor: AdmissionSimilarityAssessor = field()
    """Summarizes statistics and make sense of match data."""

    def _decorate(self, doc: FeatureDocument):
        self.doc_parser.decorate(doc)
        sent: FeatureSentence
        for sent in doc.sents:
            sent.tokens = tuple(filter(lambda t: hasattr(t, 'embedding'),
                                       sent.token_iter()))

    def _match_to_span(self, note: MatchedNote, hadm_id: str) -> FeatureSpan:
        doc: FeatureDocument = note.doc
        span: FeatureSpan = doc[note.span]
        if note.span_text != span.text:
            raise AdmissionError(
                hadm_id, f'Span mismatch: <{note.span_text}> != <{span.text}>')
        self._decorate(doc)
        return doc[note.span].to_sentence()

    def iterate_annotations(self, hadm_id: str) -> Iterable[NoteMatch]:
        stash: Stash = self.admission_similarity_assessor.match_ann_stash
        mas: MatchedAnnotationSet = stash[hadm_id]
        match: MatchedAnnotation
        for match in mas.matches:
            ds_note: MatchedNote = match.discharge_summary
            ds_span: FeatureSpan = self._match_to_span(ds_note, hadm_id)
            ant_note: MatchedNote
            for ant_note in match.note_antecedents:
                ant_span = self._match_to_span(ant_note, hadm_id)
                yield NoteMatch(
                    hadm_id=hadm_id,
                    comment_id=ds_note.cid,
                    discharge_summary_doc=ds_note.doc,
                    antecedent_doc=ant_note.doc,
                    discharge_summary_span_gold=ds_span,
                    antecedent_span_gold=ant_span)

    def predict(self, note_match: NoteMatch) -> Optional[Match]:
        res: MatchResult = self.matcher.match(
            note_match.source_doc,
            note_match.target_doc)
        if len(res.matches) > 0:
            return res.matches[0]

    def get_admission_evaluations(self, hadm_id: str) -> Iterable[Match]:
        return map(self.predict, self.iterate_annotations(hadm_id))
