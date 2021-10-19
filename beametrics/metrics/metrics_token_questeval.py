from typing import List, Dict
from beametrics.metrics.metrics import MetricBase
from questeval.token_questeval import Token_QuestEval

class MetricTokenQuestEval(MetricBase):
    def __init__(
        self,
        lang: str,
        task: str,
        *args, **kwargs
    ):
        self.metric = Token_QuestEval(language=lang, task=task)

    @classmethod
    def metric_name(cls):
        return 'token_questeval'

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

class MetricTokenQuestEval_mask_src(MetricBase):
    def __init__(
        self,
        lang: str,
        task: str,
        *args, **kwargs
    ):
        self.metric = Token_QuestEval(language=lang, task=task, doc_types=('mask_src',))

    @classmethod
    def metric_name(cls):
        return 'token_questeval_mask_src'

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

class MetricTokenQuestEval_mask_hyp(MetricBase):
    def __init__(
        self,
        lang: str,
        task: str,
        *args, **kwargs
    ):
        self.metric = Token_QuestEval(language=lang, task=task, doc_types=('mask_hyp',))

    @classmethod
    def metric_name(cls):
        return 'token_questeval_mask_hyp'

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