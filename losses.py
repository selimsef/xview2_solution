import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import NLLLoss2d


def dice_round(preds, trues, t=0.5):
    preds = (preds > t).float()
    return 1 - soft_dice_loss(preds, trues, reduce=True)


def jaccard_round(preds, trues, t=0.5, per_image=False):
    preds = (preds > t).float()
    return 1 - jaccard(preds, trues, per_image=per_image)


def soft_dice_loss(outputs, targets, per_image=False, reduce=True, ohpm=False, ohpm_pixels=256 * 256):
    batch_size = outputs.size()[0]
    eps = 1e-3
    if not per_image:
        batch_size = 1
    if ohpm:
        dice_target = targets.contiguous().view(-1).float()
        dice_output = outputs.contiguous().view(-1)

        loss_b = torch.abs(dice_target - dice_output)
        _, indc = loss_b.topk(ohpm_pixels)
        dice_target = dice_target[indc]
        dice_output = dice_output[indc]
        intersection = torch.sum(dice_output * dice_target)
        union = torch.sum(dice_output) + torch.sum(dice_target) + eps
        loss = (1 - (2 * intersection + eps) / union)
        loss = loss.mean()
    else:
        dice_target = targets.contiguous().view(batch_size, -1).float()
        dice_output = outputs.contiguous().view(batch_size, -1)
        intersection = torch.sum(dice_output * dice_target, dim=1)
        union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
        loss = (1 - (2 * intersection + eps) / union)
        if reduce:
            loss = loss.mean()

    return loss


def jaccard(outputs, targets, per_image=False, non_empty=False, min_pixels=5):
    batch_size = outputs.size()[0]
    eps = 1e-3
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    target_sum = torch.sum(dice_target, dim=1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    losses = 1 - (intersection + eps) / (torch.sum(dice_output + dice_target, dim=1) - intersection + eps)
    if non_empty:
        assert per_image == True
        non_empty_images = 0
        sum_loss = 0
        for i in range(batch_size):
            if target_sum[i] > min_pixels:
                sum_loss += losses[i]
                non_empty_images += 1
        if non_empty_images == 0:
            return 0
        else:
            return sum_loss / non_empty_images

    return losses.mean()


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False, ohpm=False, ohpm_pixels=256 * 256):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image
        self.ohpm = ohpm
        self.ohpm_pixels = ohpm_pixels

    def forward(self, input, target):
        return soft_dice_loss(input, target, per_image=self.per_image, ohpm=self.ohpm, ohpm_pixels=self.ohpm_pixels)


class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False, non_empty=False, apply_sigmoid=False,
                 min_pixels=5):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image
        self.non_empty = non_empty
        self.apply_sigmoid = apply_sigmoid
        self.min_pixels = min_pixels

    def forward(self, input, target):
        if self.apply_sigmoid:
            input = torch.sigmoid(input)
        return jaccard(input, target, per_image=self.per_image, non_empty=self.non_empty, min_pixels=self.min_pixels)


class StableBCELoss(nn.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        return bce_loss(input, target).mean()


def bce_loss(input, target):
    input = input.float().view(-1)
    target = target.float().view(-1)
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss


def bce_loss_sigmoid(input, target):
    eps = 1e-6
    outputs = torch.clamp(input, eps, 1. - eps)
    targets = torch.clamp(target, eps, 1. - eps)
    pt = (1 - targets) * (1 - outputs) + targets * outputs
    return -torch.log(pt)


class ComboLoss(nn.Module):
    def __init__(self, weights, per_image=False, only_target_pixels=False, skip_empty=False,
                 channel_weights=np.ones((20,)), channel_losses=None,
                 ohpm=False, ohpm_pixels=100 * 100, reduce='sum'):
        super().__init__()
        self.weights = weights
        self.bce = StableBCELoss()
        self.dice = DiceLoss(per_image=per_image, ohpm=ohpm, ohpm_pixels=ohpm_pixels)
        self.jaccard = JaccardLoss(per_image=per_image)
        self.focal = FocalLoss2d()
        self.mapping = {'bce': self.bce,
                        'dice': self.dice,
                        'focal': self.focal,
                        'jaccard': self.jaccard}
        self.expect_sigmoid = {'dice', 'focal', 'jaccard'}
        self.per_channel = {'dice', 'jaccard'}
        self.values = {}
        self.channel_weights = channel_weights
        self.channel_losses = channel_losses
        self.skip_empty = skip_empty
        self.only_target_pixels = only_target_pixels
        self.reduce = reduce

    def forward(self, outputs, targets):
        loss = 0
        weights = self.weights
        sigmoid_input = torch.sigmoid(outputs)
        original_sigmoid = sigmoid_input
        if self.only_target_pixels:
            sigmoid_input = sigmoid_input * targets
        for k, v in weights.items():
            if not v:
                continue
            val = 0
            if k in self.per_channel:
                channels = targets.size(1)
                val_channels = []
                for c in range(channels):
                    if not self.channel_losses or k in self.channel_losses[c]:
                        if self.skip_empty and torch.sum(targets[:, c, ...]) < 50 and torch.sum(
                                sigmoid_input[:, c, ...]) < 50:
                            continue
                        val_channels.append(self.channel_weights[c] * self.mapping[k](
                            sigmoid_input[:, c, ...] if k in self.expect_sigmoid else outputs[:, c, ...],
                            targets[:, c, ...]))
                if self.reduce == 'avg':
                    val_channels = sum(val_channels) / channels
                elif self.reduce == 'harmonic_mean':
                    eps = 1e-4
                    val_channels = 1 - channels / sum([1 / (1 - v - eps) for v in val_channels])
                elif self.reduce == 'sum':
                    val_channels = sum(val_channels)
                else:
                    raise NotImplementedError(self.reduce + " is not implemented")
                val += val_channels
            else:
                val = self.mapping[k](original_sigmoid if k in self.expect_sigmoid else outputs, targets)

            self.values[k] = val
            loss += self.weights[k] * val
        return loss


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        eps = 1e-5
        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1. - eps)
        targets = torch.clamp(targets, eps, 1. - eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1. - pt) ** self.gamma * torch.log(pt)).mean()


class FocalLossWithDice(nn.Module):
    def __init__(self, num_classes, ignore_index=255, gamma=2, ce_weight=1., d_weight=0.1, weight=None,
                 size_average=True, ohpm=False, ohpm_pixels=128 * 128):
        super().__init__()
        self.num_classes = num_classes
        self.d_weight = d_weight
        self.ce_w = ce_weight
        self.gamma = gamma
        if weight is not None:
            weight = torch.Tensor(weight).float()
        self.nll_loss = NLLLoss2d(weight, size_average, ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.ohpm = ohpm
        self.ohpm_pixels = ohpm_pixels

    def forward(self, outputs, targets):
        probas = F.softmax(outputs, dim=1)
        ce_loss = self.nll_loss((1 - probas) ** self.gamma * F.log_softmax(outputs, dim=1), targets)
        d_loss = soft_dice_loss_mc(outputs, targets, self.num_classes, ignore_index=self.ignore_index, ohpm=self.ohpm,
                                   ohpm_pixels=self.ohpm_pixels)
        non_ignored = targets != 255
        loc = soft_dice_loss(1 - probas[:, 0, ...][non_ignored], (targets[non_ignored] > 0) * 1.)

        return self.ce_w * ce_loss + self.d_weight * d_loss + self.d_weight * loc


def soft_dice_loss_mc(outputs, targets, num_classes, per_image=False, only_existing_classes=False, ignore_index=255,
                      minimum_class_pixels=10, reduce_batch=True, ohpm=True, ohpm_pixels=16384):
    batch_size = outputs.size()[0]
    eps = 1e-5
    outputs = F.softmax(outputs, dim=1)

    def _soft_dice_loss(outputs, targets):
        loss = 0
        non_empty_classes = 0
        for cls in range(1, num_classes):
            non_ignored = targets.view(-1) != ignore_index
            dice_target = (targets.view(-1)[non_ignored] == cls).float()
            dice_output = outputs[:, cls].contiguous().view(-1)[non_ignored]
            if ohpm:
                loss_b = torch.abs(dice_target - dice_output)
                px, indc = loss_b.topk(ohpm_pixels)
                dice_target = dice_target[indc]
                dice_output = dice_output[indc]

            intersection = (dice_output * dice_target).sum()
            if dice_target.sum() > minimum_class_pixels:
                union = dice_output.sum() + dice_target.sum() + eps
                loss += (1 - (2 * intersection + eps) / union)
                non_empty_classes += 1
        if only_existing_classes:
            loss /= (non_empty_classes + eps)
        else:
            loss /= (num_classes - 1)
        return loss

    if per_image:
        if reduce_batch:
            loss = 0
            for i in range(batch_size):
                loss += _soft_dice_loss(torch.unsqueeze(outputs[i], 0), torch.unsqueeze(targets[i], 0))
            loss /= batch_size
        else:
            loss = torch.Tensor(
                [_soft_dice_loss(torch.unsqueeze(outputs[i], 0), torch.unsqueeze(targets[i], 0)) for i in
                 range(batch_size)])
    else:
        loss = _soft_dice_loss(outputs, targets)

    return loss




