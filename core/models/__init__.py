from .backbone import BERT, iresnet100, TCN, ViT, \
                        MultiBERT, MultiBERTV2, MultiBERTV3, Multi_TBERT, \
                        GRU, LSTM, BiGRU, BERT_BiGRU, BERT_BiGRUV2, \
                        TCN_BERTV2, BiLSTM, TCN_BiGRU, TCN_BiGRUV2
from .loss import BCELoss, MultiLabelSoftMarginLoss, WeightedAsymmetricLoss, RDropLoss
from .metric import AUMetric
from .optimizer import build_optimizer, build_lr_scheduler

from ..utils import  MODEL_REGISTRY, OPTIMIZER_REGISTRY, \
                        METRIC_REGISTRY, LOSS_REGISTRY


def build_model(cfg):
    return MODEL_REGISTRY.build(cfg)


def build_loss(cfg):
    return LOSS_REGISTRY.build(cfg)


def build_metric(cfg):
    return METRIC_REGISTRY.build(cfg)