from typing import List, Dict
from beametrics.metrics.metrics import MetricBase
from questeval.questeval_metric import QuestEval
from questeval.mask_questeval import Mask_QuestEval, Mask_QuestEval_src, Mask_QuestEval_hyp

class MetricQuestEval(MetricBase):
    def __init__(
        self,
        lang: str,
        task: str,
        *args, **kwargs
    ):
        self.metric = QuestEval(language=lang, task=task)

    @classmethod
    def metric_name(cls):
        return 'questeval'

    def pipeline(
            self,
            predictions: List[str] = None,
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    ) -> Dict:
        res = self.metric.corpus_questeval(
                hypothesis = predictions,
                sources = sources,
                list_references = list_references
        )

        return {self.metric_name(): res['ex_level_scores']}

class MetricMaskQuestEval(MetricBase):
    def __init__(
        self,
        lang: str,
        task: str,
        *args, **kwargs
    ):
        self.metric = Mask_QuestEval(language=lang, task=task)

    @classmethod
    def metric_name(cls):
        return 'mask_questeval'

    def pipeline(
            self,
            predictions: List[str] = None,
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    ) -> Dict:
        res = self.metric.corpus_questeval(
                hypothesis = predictions,
                sources = sources,
                list_references = list_references
        )

        return {self.metric_name(): res['ex_level_scores']}

class MetricMaskQuestEval_hyp(MetricBase):
    def __init__(
        self,
        lang: str,
        task: str,
        *args, **kwargs
    ):
        self.metric = Mask_QuestEval_hyp(language=lang, task=task)

    @classmethod
    def metric_name(cls):
        return 'mask_questeval_hyp'

    def pipeline(
            self,
            predictions: List[str] = None,
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    ) -> Dict:
        res = self.metric.corpus_questeval(
                hypothesis = predictions,
                sources = sources,
                list_references = list_references
        )

        return {self.metric_name(): res['ex_level_scores']}

class MetricMaskQuestEval_src(MetricBase):
    def __init__(
        self,
        lang: str,
        task: str,
        *args, **kwargs
    ):
        self.metric = Mask_QuestEval_src(language=lang, task=task)

    @classmethod
    def metric_name(cls):
        return 'mask_questeval_src'

    def pipeline(
            self,
            predictions: List[str] = None,
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    ) -> Dict:
        res = self.metric.corpus_questeval(
                hypothesis = predictions,
                sources = sources,
                list_references = list_references
        )

        return {self.metric_name(): res['ex_level_scores']}