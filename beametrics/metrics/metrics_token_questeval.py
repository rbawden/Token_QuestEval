from typing import List, Dict
from beametrics.metrics.metrics import MetricBase
from questeval.token_questeval import Token_QuestEval, Token_QuestEval_src, Token_QuestEval_hyp

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

class MetricTokenQuestEval_hyp(MetricBase):
    def __init__(
        self,
        lang: str,
        task: str,
        *args, **kwargs
    ):
        self.metric = Token_QuestEval_hyp(language=lang, task=task)

    @classmethod
    def metric_name(cls):
        return 'token_questeval_hyp'

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

class MetricTokenQuestEval_src(MetricBase):
    def __init__(
        self,
        lang: str,
        task: str,
        *args, **kwargs
    ):
        self.metric = Token_QuestEval_src(language=lang, task=task)

    @classmethod
    def metric_name(cls):
        return 'token_questeval_src'

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