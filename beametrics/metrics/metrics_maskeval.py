from typing import List, Dict
from beametrics.metrics.metrics import MetricBase
from questeval.maskeval import MaskEval

class MetricMaskEval(MetricBase):
    def __init__(
        self,
        lang: str,
        task: str,
        *args, **kwargs
    ):
        self.metric = MaskEval(language=lang)

    @classmethod
    def metric_name(cls):
        return 'maskeval'

    def pipeline(
            self,
            predictions: List[str] = None,
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    ) -> Dict:
        res = self.metric.corpus_questeval(
                hypothesis = predictions,
                references = sources,
        )

        return {self.metric_name(): res['ex_level_scores']}

class MetricMaskEval_mask_src(MetricBase):
    def __init__(
        self,
        lang: str,
        task: str,
        *args, **kwargs
    ):
        self.metric = MaskEval(language=lang, mask_types_to_consider = ("ref",))

    @classmethod
    def metric_name(cls):
        return 'maskeval_mask_src'

    def pipeline(
            self,
            predictions: List[str] = None,
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    ) -> Dict:
        res = self.metric.corpus_questeval(
            hypothesis=predictions,
            references=sources,
        )

        return {self.metric_name(): res['ex_level_scores']}

class MetricMaskEval_mask_hyp(MetricBase):
    def __init__(
        self,
        lang: str,
        task: str,
        *args, **kwargs
    ):
        self.metric = MaskEval(language=lang, mask_types_to_consider = ("hyp",))

    @classmethod
    def metric_name(cls):
        return 'maskeval_mask_hyp'

    def pipeline(
            self,
            predictions: List[str] = None,
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    ) -> Dict:
        res = self.metric.corpus_questeval(
            hypothesis=predictions,
            references=sources,
        )

        return {self.metric_name(): res['ex_level_scores']}