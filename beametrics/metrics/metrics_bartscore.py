from typing import List, Dict
from beametrics.metrics.metrics import MetricBase
from bartscore.bartscore_metric import BARTScore, BARTScore_tgt_hyp, BARTScore_tgt_src

class MetricBartscore(MetricBase):
    def __init__(
        self,
        lang: str,
        task: str,
        *args, **kwargs
    ):
        self.metric = BARTScore()

    @classmethod
    def metric_name(cls):
        return 'bartscore'

    def pipeline(
            self,
            predictions: List[str] = None,
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    ) -> Dict:
        res = self.metric.corpus_score(
                hypothesis = predictions,
                sources = sources,
        )

        return {self.metric_name(): res}

class MetricBartscore_tgt_src(MetricBase):
    def __init__(
        self,
        lang: str,
        task: str,
        *args, **kwargs
    ):
        self.metric = BARTScore_tgt_src()

    @classmethod
    def metric_name(cls):
        return 'bartscore_tgt_src'

    def pipeline(
            self,
            predictions: List[str] = None,
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    ) -> Dict:
        res = self.metric.corpus_score(
                hypothesis = predictions,
                sources = sources,
        )

        return {self.metric_name(): res}

class MetricBartscore_tgt_hyp(MetricBase):
    def __init__(
        self,
        lang: str,
        task: str,
        *args, **kwargs
    ):
        self.metric = BARTScore_tgt_hyp()

    @classmethod
    def metric_name(cls):
        return 'bartscore_tgt_hyp'

    def pipeline(
            self,
            predictions: List[str] = None,
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    ) -> Dict:
        res = self.metric.corpus_score(
                hypothesis = predictions,
                sources = sources,
        )

        return {self.metric_name(): res}