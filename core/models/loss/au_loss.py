import torch
import torch.nn.functional as F
import torch.nn as nn

from core.utils import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class MultiLabelSoftMarginLoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        pw = torch.FloatTensor(pos_weight)
        self.multi_label_loss = nn.MultiLabelSoftMarginLoss(weight=pw)

    def forward(self, x, y):
        return self.multi_label_loss(x, y)


@LOSS_REGISTRY.register()
class BCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        pw = torch.FloatTensor(pos_weight)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, x, y):
        return self.bce(x, y)


@LOSS_REGISTRY.register()
class RDropLoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        pw = torch.FloatTensor(pos_weight)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits1, logits2, gt):
        loss = (self.bce(logits1, gt) + self.bce(logits2, gt)) / 2

        return loss


@LOSS_REGISTRY.register()
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, gt):
        ce_loss = F.cross_entropy(pred, gt, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)** self.gamma * ce_loss).mean()

        return focal_loss


class SoftTarget(nn.Module):
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T


	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, inputs, target):
        target = F.softmax(target, dim=-1)
        logprobs = torch.nn.functional.log_softmax(inputs.view(inputs.shape[0], -1), dim=1)
        batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
        return torch.mean(batchloss)


@LOSS_REGISTRY.register()
class RDropLoss(nn.Module):
    def __init__(self, weight, alpha=5):
        super().__init__()
        w = torch.FloatTensor(weight)
        self.ce = nn.CrossEntropyLoss(weight=w, reduction='none')
        self.kld = nn.KLDivLoss(reduction='none')
        self.alpha = alpha

    def forward(self, logits1, logits2, gt):
        ce_loss = (self.ce(logits1, gt) + self.ce(logits2, gt)) / 2
        kl_loss1 = self.kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1)).sum(-1)
        kl_loss2 = self.kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1)).sum(-1)
        kl_loss = (kl_loss1 + kl_loss2) / 2
        loss = ce_loss + self.alpha * kl_loss

        loss = loss.mean(-1)

        return loss

@LOSS_REGISTRY.register()
class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, eps=1e-8, disable_torch_grad=True, weight=None):
        super(WeightedAsymmetricLoss, self).__init__()
        self.disable_torch_grad = disable_torch_grad
        self.eps = eps

        self.local_rank = torch.distributed.get_rank()
        self.weight = torch.tensor(weight).cuda(self.local_rank)

    def forward(self, x, y):

        xs_pos = x
        xs_neg = 1 - x

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.disable_torch_grad:
            torch.set_grad_enabled(False)
        neg_weight = 1 - xs_neg
        if self.disable_torch_grad:
            torch.set_grad_enabled(True)
        loss = los_pos + neg_weight * los_neg

        if self.weight is not None:
            loss = loss * self.weight.view(1,-1)

        loss = loss.mean(dim=-1)
        return -loss.mean()

