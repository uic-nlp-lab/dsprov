"""Extract the annotations to JSON.

"""
__author__ = 'Paul Landes'

from typing import List, Dict, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import sys
import itertools as it
from zensols.config import Dictable
from . import (
    MatchedSection, MatchedNote, MatchedAnnotation, MatchedAnnotationSet,
    MatchAnnotationSetStash
)


@dataclass
class AnnotationExtractor(Dictable):
    """Extracts the human annotations an in memory Python tree, JSON or YAML
    using the superclass interface.

    """
    match_ann_stash: MatchAnnotationSetStash = field(repr=False)
    """Creates instances of :class:`.MatchedAnnotationSet`."""

    discharge_summary_key: str = field(default='ds')
    """The discharge summary key used in the flatted data structure."""

    note_antecedent_summary_key: str = field(default='ant')
    """The note antecedent key used in the flatted data structure."""

    include_annotator: bool = field(default=False)
    """Whether to add the annotator identifier."""

    include_text: bool = field(default=False)
    """Whether to add the annotation's text."""

    include_sections: bool = field(default=False)
    """Whether to add matched sections.  There will be more than one when
    annotations cross section boundaries.

    """
    limit: int = field(default=sys.maxsize)
    """The limit on the number of admissions to output."""

    def _matched_section(self, ms: MatchedSection):
        dct = OrderedDict(
            [['id', ms.section_id],
             ['header_span', tuple(map(lambda s: s.asdict(), ms.header_spans))],
             ['body_span', ms.body_span.asdict()]])
        if self.include_text:
            dct['text'] = ms.matched_text
        return dct

    def _matched_note(self, mn: MatchedNote, is_ds: bool) -> Dict[str, Any]:
        dct = OrderedDict(
            [['note_span', mn.span.asflatdict()]])
        if self.include_sections:
            dct['sections'] = tuple(map(self._matched_section, mn.sections))
        if not is_ds:
            dct['row_id'] = mn.row_id
            dct['note_category'] = mn.note_category
        if self.include_text:
            dct['text'] = mn.span_text
        return dct

    def _match(self, ma: MatchedAnnotation):
        ants = []
        ds_match: MatchedNote = ma.discharge_summary
        ant_match: MatchedNote
        for ant_match in ma.note_antecedents:
            assert ds_match.cid == ant_match.cid
            ants.append(self._matched_note(ant_match, False))
        return OrderedDict(
            [[self.discharge_summary_key, self._matched_note(ds_match, True)],
             [self.note_antecedent_summary_key, ants]])

    def _matched_anon_set(self, mas: MatchedAnnotationSet):
        matches: List[Dict[str, Any]] = {}
        ma: MatchedAnnotation
        for ma in mas.matches:
            matches[int(ma.discharge_summary.cid)] = self._match(ma)
        dct = OrderedDict(
            [[f'{self.discharge_summary_key}_row_id',
              self._matched_note(mas.matches[0].discharge_summary, False)],
             ['matches', matches]])
        if self.include_annotator:
            dct['annotator'] = ma.annotator
        return dct

    def _from_dictable(self, *args, **kwargs):
        adms: Dict[str, Dict[str, Any]] = OrderedDict()
        mas: MatchedAnnotationSet
        for mas in it.islice(self.match_ann_stash.values(), self.limit):
            adms[int(mas.hadm_id)] = self._matched_anon_set(mas)
        return adms
