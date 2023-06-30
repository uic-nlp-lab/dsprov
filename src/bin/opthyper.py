#!/usr/bin/env python

"""Hyperparameter optimization for the discharge summary provenience of data
unsupervised model.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Tuple, Dict, List, Iterable, Optional, Set
from dataclasses import dataclass, field
import sys
import logging; logging.basicConfig(level=logging.WARNING)
import re
from pathlib import Path
from io import TextIOBase
import itertools as it
import plac
import pandas as pd
from hyperopt import hp
from zensols.config import ConfigFactory
from zensols.cli import CliHarness; CliHarness.add_sys_path(Path('src/python'))
from zensols.nlp import FeatureSpan
from zensols.nlp.score import ScoreContext, ScoreSet
from zensols.datdesc import HyperparamModel, DataDescriber
from zensols.datdesc.optscore import ScoringHyperparameterOptimizer
from zensols.spanmatch import Match
from zensols.datdesc.opt import HyperparamResult
from uic.dsprov.pred import PredictionMatcher, NoteMatch
from uic.dsprov import AdmissionSimilarityAssessor

logger = logging.getLogger(__name__)


@dataclass
class DSProvHyperOptimizer(ScoringHyperparameterOptimizer):
    """The hyperparameter optimizer for discharge summary matching.

    """
    hadm_ids: Tuple[str] = field(default=None)
    """The admission IDs in the optimiziation."""

    hadm_limit: int = field(default=sys.maxsize)
    """The max number of admissions (not used when :obj:`hadm_ids` is set). """

    match_sample_size: Optional[int] = field(default=None)
    """The number of match samples to score for each cycle, or the size the
    entire match set if ``None``.

    """
    def __post_init__(self):
        super().__post_init__()
        self._anon_iter = None

    @property
    def matcher(self) -> PredictionMatcher:
        return self.config_factory('dsprov_pred_matcher')

    @property
    def assessor(self) -> AdmissionSimilarityAssessor:
        return self.config_factory('dsprov_admission_similarity_assessor')

    def _create_config_factory(self) -> ConfigFactory:
        logger.setLevel(logging.INFO)
        logging.getLogger('uic.dsprov.pred').setLevel(logging.WARNING)
        logging.getLogger('zensols.datdesc.opt').setLevel(logging.INFO)
        harness = CliHarness(
            app_factory_class='uic.dsprov.ApplicationFactory')
        return harness.get_config_factory()

    def _get_hyperparams(self) -> HyperparamModel:
        return self.matcher.matcher.hyp

    def _get_hadm_ids(self) -> Tuple[str]:
        ids: Tuple[str] = self.hadm_ids
        if ids is None:
            ids = sorted(self.assessor.hadm_ids)
            first = '120312'
            ids.remove(first)
            ids.insert(0, first)
            if self.hadm_limit is not None:
                ids = ids[:self.hadm_limit]
        return ids

    def _get_match_count(self) -> int:
        df: pd.DataFrame = self.assessor.match_dataframe
        df = df['hadm_id ds_row_id ant_row_id comment_id'.split()]
        return len(df.drop_duplicates())

    def _get_matches(self) -> Iterable[NoteMatch]:
        matcher: PredictionMatcher = self.matcher
        hadm_id: str
        for hadm_id in self._get_hadm_ids():
            matches: Iterable[NoteMatch] = matcher.iterate_annotations(hadm_id)
            note_match: NoteMatch
            for note_match in matches:
                yield note_match

    def _get_next_matches(self, reset: bool = False) -> Iterable[NoteMatch]:
        sample_size: int = self.match_sample_size
        if sample_size is None:
            sample_size = self._get_match_count()
        if reset or self._anon_iter is None:
            self._anon_iter = it.cycle(self._get_matches())
        if not reset:
            return it.islice(self._anon_iter, sample_size)

    def _get_next_score_context(self) -> ScoreContext:
        matcher: PredictionMatcher = self.matcher
        pairs: List[Tuple[FeatureSpan, FeatureSpan]] = []
        cor_ids: List[str] = []
        note_match: NoteMatch
        for note_match in self._get_next_matches():
            match: Match = matcher.predict(note_match)
            pairs.extend([[note_match.source_span, match.source_span],
                          [note_match.target_span, match.target_span]])
            cor_ids.extend(map(lambda n: f'{note_match.key}.{n[0]}',
                               NoteMatch.source_target_names()))
        return ScoreContext(pairs=pairs, correlation_ids=cor_ids)

    def _get_loss(self, df: pd.DataFrame) -> float:
        return float(1 - df['semeval_partial_f_score'].mean())

    def _get_score_dataframe(self, score_set: ScoreSet) -> pd.DataFrame:
        df: pd.DataFrame = score_set.as_dataframe()
        df.insert(1, 'note',
                  list(it.islice(it.cycle(NoteMatch.source_target_names()),
                                 len(df))))
        return df

    def _create_space(self) -> Dict[str, float]:
        ss_params = (
            ('source_distance_threshold', 0.3, 10),
            ('source_position_scale', 0.8, 60),
            ('target_distance_threshold', 0.3, 10),
            ('target_position_scale', 0.8, 60))
        space = self._create_uniform_space(ss_params)
        if 0:
            space |= self._create_uniform_space(
                (('min_source_token_span', 1, 5),
                 ('min_target_token_span', 1, 5)),
                integer=True)
            space['cased'] = hp.choice('cased', [True, False])
        return space

    def _write_result(self, res: HyperparamResult, writer: TextIOBase):
        super()._write_result(res, writer)
        df = res.scores
        writer.write('averages:\n')
        for name in df['note'].drop_duplicates():
            dfn = df[df['note'] == name]
            score: float = dfn['semeval_partial_f_score'].mean()
            print(f'  {name}: {score}', file=writer)

    def _compact_columns(self, df: pd.DataFrame,
                         rename: bool = False) -> pd.DataFrame:
        if 0:
            cols = list(filter(lambda n: re.match(r'.*f_score$', n) is not None,
                               df.columns))
            cols.append('exact_match')
            df = df[cols]
        col_rep = dict(map(lambda n:
                           (n, re.sub(r'^([sr])(?:emeval|ouge)', r'\1', n)),
                           df.columns))
        col_rep['exact_match'] = 'em'
        if rename:
            df = df.rename(columns=col_rep)
        return df

    def summarize_agg_scores(self) -> DataDescriber:
        """Create a data describer with the aggregate results of testing wtih
        the optimal hyperparams ready to be saved.

        """
        def summarize(note: str, df: pd.DataFrame) -> pd.DataFrame:
            df: pd.DataFrame = self._compact_columns(
                df[df['note'] == note].
                groupby('name').agg('mean').
                sort_values('semeval_partial_f_score'))
            df = df.drop(columns=('semeval_ent_type_f_score ' +
                                  'semeval_strict_f_score ' +
                                  'semeval_exact_f_score').split())
            df.insert(0, 'name', df.index)
            df = df.reset_index(drop=True)
            return df

        df: pd.DataFrame = self.gather_aggregate_scores()
        counts_by_name: Dict[str, int] = dict(map(
            lambda x: (x[0], len(x[1])), df.groupby('name')))
        count_vals: Set[int] = set(counts_by_name.values())
        assert len(count_vals) == 1
        counts: int = next(iter(count_vals))
        df_ds: pd.DataFrame = summarize('discharge_summary', df)
        df_ant: pd.DataFrame = summarize('antecedent', df)
        dd_ds = self.config_factory.new_instance(
            name='dsprov_tuned_dataframe_describer',
            df=df_ds,
            desc=f'Discharge Summary Match Scores on {counts} Mathces',
        )
        dd_ds.name = 'tuned_discharge_summary'
        dd_da = self.config_factory.new_instance(
            name='dsprov_tuned_dataframe_describer',
            df=df_ant,
            desc=f'Note Antecedent Match Scores on {counts} Mathces',
        )
        dd_da.name = 'tuned_antecedent'
        return self.config_factory.new_instance(
            name='dsprov_tuned_data_describer',
            describers=(dd_ds, dd_da))

    def write_agg_scores(self):
        """Write aggregate results of testing wtih the optimal hyperparams along
        with the YAML config files.

        """
        dd = self.summarize_agg_scores()
        dd.save()


@plac.annotations(
    action=('action', 'positional', None, str,
            'opt best score compare score scores agg dumpagg'.split()),
    evals=('number of evaluations', 'option', 'e'),
    sample=('the number of match samples to score each cycle', 'option', 's'),
    adm=('the max number of admissions', 'option', 'a'),
    name=('the name of the experimentation run', 'option', 'n'),
    reverse=('reverse discharge vs antecedent note order', 'flag', 'r'),
    baseline=('a JSON file of hyperparameter settings to set on start',
              'option', 'b'))
def main(action: str, evals: int = 1, sample: int = 0, reverse: bool = False,
         adm: int = 0, name: str = 'default', baseline: str = '-'):
    if reverse:
        NoteMatch.reverse()
    optimizer = DSProvHyperOptimizer(
        max_evals=evals,
        match_sample_size=sample if sample > 0 else None,
        hadm_limit=adm if adm > 0 else None,
        name=name,
        baseline_path=None if baseline == '-' else Path(baseline))
    if action is not None:
        {'opt': optimizer.optimize,
         'best': lambda: optimizer.write_best_result(include_param_json=True),
         'compare': optimizer.write_compare,
         'score': optimizer.write_score,
         'scores': lambda: optimizer.write_scores(iterations=evals),
         'agg': optimizer.aggregate_scores,
         'dumpagg': optimizer.write_agg_scores,
         }[action]()
    return optimizer


if (__name__ == '__main__'):
    plac.call(main)
