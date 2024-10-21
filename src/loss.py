import torch
import torch.nn as nn
from torch.autograd import Variable

class BCELossWeightInLog(nn.Module):
    """Add weights in log L = -plog(wp) - (1-p)log((1-w)(1-p))
    - focus: pos. L = -plog(wp) - (1-p)log(1-p)
    - focus: neg. L = -plog(p) - (1-p)log((1-w)(1-p))
    """
    def __init__(self, focus=None):
        super(BCELossWeightInLog, self).__init__()
        self.focus = focus
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions:torch.tensor, targets:torch.tensor, weight:torch.tensor):
        assert torch.all(torch.logical_and(weight >=0, weight <=1)), "Weights are not in range [0, 1]"
        targets = targets.float()
        if self.focus is None:
            loss = -targets * torch.log(weight * predictions + 1e-15) - (1 - targets) * torch.log((1-weight) * (1 - predictions) + 1e-15)
        elif self.focus == 'neg':
            loss = -targets * torch.log(predictions + 1e-15) - (1 - targets) * torch.log((1-weight) * (1 - predictions) + 1e-15)
        elif self.focus == 'pos':
            loss = -targets * torch.log(weight * predictions + 1e-15) - (1 - targets) * torch.log(1 - predictions + 1e-15)
        return torch.mean(loss)

class BCELossWeightOutLog(nn.Module):
    """Add weights in log L = -wplog(p) + w(1-p)log(1-p))
    - focus: pos. L = -gamma*w*plog(p) + (1-p)log(1-p)
    - focus: neg. L = -plog(p) - gamma*(1-w)*(1-p)log(1-p)
    """
    def __init__(self, focus=None, gamma=1):
        super(BCELossWeightOutLog, self).__init__()
        self.focus = focus
        self.gamma = gamma

    def forward(self, predictions, targets, weight):
        assert torch.all(torch.logical_and(weight >=0, weight <=1)), "Weights are not in range [0, 1]"
        targets = targets.float()
        if self.focus is None:
            loss = -targets * (1-weight) * torch.log(predictions + 1e-15) - (1 - targets) * weight * torch.log(1 - predictions + 1e-15)
        elif self.focus == 'neg':
            loss = -targets * torch.log(predictions + 1e-15) - self.gamma * weight * (1 - targets) * torch.log(1 - predictions + 1e-15)
        elif self.focus == 'pos':
            loss = -self.gamma * (1-weight) *  targets * torch.log(predictions + 1e-15) - (1 - targets) * torch.log(1 - predictions + 1e-15)

        return torch.mean(loss)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device('cuda')
                  
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1) # B,2,D

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1) #B, 1
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device) # B,B diagonal = 1.0
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1] # n_views: 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # split B,2,D to B * 2,D
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all': # use this mode
            anchor_feature = contrast_feature # 2B,D
            anchor_count = contrast_count # 2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) # dot prod /temp  (2B, 2B)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # x.XT - c (c not in backward)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count) # 2B,2B
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        ) # matrix ones 2B,2B with diagonal is zero
        mask = mask * logits_mask # find others positive those are not query

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask # take exp except query itself
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class NConLoss(nn.Module):
    def __init__(
            self,
            temperature=0.07,
            base_temperature=0.07,
            beta=1,
            estimator='hard', 
            ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.estimator = estimator
        self.beta = beta

    def forward(self, query_feature, features, labels=None):
        # neg score
        # out = torch.cat([out_1, out_2], dim=0)
        # check the shape of features
        # query_feature: D (detach, use as a constant)
        # features: B,D
        # labels: B 1/0
        
        features = features.view(features.shape[0], -1)
        labels = labels.float()

        anchor_feature = features

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, query_feature),
            self.temperature)

        logits_max = anchor_dot_contrast.max()
        logits = anchor_dot_contrast - logits_max.detach()

        positive_mask = labels
        negative_mask = 1 - positive_mask

        exp_logits = torch.exp(logits)
        exp_neg_logits = exp_logits * negative_mask
        
        pos = (exp_logits * positive_mask).sum()

        if self.estimator == 'hard':
            # this number should be the number of negative samples
            b = self.beta * exp_neg_logits * negative_mask.sum() / exp_neg_logits.sum() 
            neg = (b * exp_neg_logits).sum()
        elif self.estimator == 'easy':
            neg = exp_neg_logits.sum()
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

        loss_temp = self.temperature / self.base_temperature
        log_prob = logits - torch.log(pos + neg + 1e-6)
        loss = - loss_temp * torch.sum(positive_mask * log_prob) / (positive_mask.sum() + 1e-6)

        return loss