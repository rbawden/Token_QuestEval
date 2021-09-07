from beametrics.metrics.metrics_hugging_face import *
from beametrics.metrics.metrics_stats import *
from beametrics.metrics.metrics_questeval import *
from beametrics.metrics.metrics_token_questeval import *
from beametrics.metrics.metrics_bartscore import *

_D_METRICS = {
    MetricLength.metric_name(): MetricLength,
    MetricAbstractness.metric_name(): MetricAbstractness,
    MetricRepetition.metric_name(): MetricRepetition,
    MetricSacreBleu.metric_name(): MetricSacreBleu,
    MetricRouge.metric_name(): MetricRouge,
    MetricMeteor.metric_name(): MetricMeteor,
    MetricSari.metric_name(): MetricSari,
    MetricBertscore.metric_name(): MetricBertscore,
    MetricBleurtScore.metric_name(): MetricBleurtScore,
    #MetricQuestEval.metric_name(): MetricQuestEval,
    MetricMaskQuestEval.metric_name(): MetricMaskQuestEval,
    MetricMaskQuestEval_hyp.metric_name(): MetricMaskQuestEval_hyp,
    MetricMaskQuestEval_src.metric_name(): MetricMaskQuestEval_src,
    MetricTokenQuestEval.metric_name(): MetricTokenQuestEval,
    MetricTokenQuestEval_hyp.metric_name(): MetricTokenQuestEval_hyp,
    MetricTokenQuestEval_src.metric_name(): MetricTokenQuestEval_src,
    MetricBartscore.metric_name(): MetricBartscore,
    MetricBartscore_tgt_src.metric_name(): MetricBartscore_tgt_src,
    MetricBartscore_tgt_hyp.metric_name(): MetricBartscore_tgt_hyp,

}