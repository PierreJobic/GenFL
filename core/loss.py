import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class loss_nll(nn.Module):
    def forward(self, output, target, pmin=None, bounded=None):
        """
        output.shape = (batch_size, num_classes) in [-inf, 0]
        target.shape = (batch_size) in {0, 1, ..., num_classes - 1}
        """
        return F.nll_loss(output, target)

    def correct(self, output, targets):
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(targets.view_as(pred)).sum().item()
        return correct


class loss_logistic(nn.Module):
    def forward(self, output, targets, pmin=None, bounded=None):
        """
        output.shape = (batch_size, num_classes) in [-inf, inf]
        targets.shape = (batch_size) in {-1, 1}
        """
        # return F.binary_cross_entropy(output, targets.view(-1, 1).float())
        targets = targets.view(-1, 1).float()
        switch = ((targets * output) < (-targets * output)).float()
        exponent = torch.exp((2 * switch - 1) * targets * output)
        res = (torch.log(1 + exponent) - switch * targets * output).mean() / math.log(2)
        return res

    def correct(self, output, targets):
        # pred = output.argmax(dim=1, keepdim=True)
        # correct = pred.eq(targets.view_as(pred)).sum().item()
        # return correct
        targets = targets.view(-1, 1).float()
        pred = (output > 0).float() - (output <= 0).float()
        correct = pred.eq(targets.view_as(pred)).sum().item()
        return correct


class loss_01(nn.Module):
    def forward(self, output, targets, pmin=None, bounded=None):
        """
        output.shape = (batch_size, num_classes)
        """
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(targets.view_as(pred)).sum().item()
        total = targets.size(0)
        res_01 = 1 - (correct / total)
        return res_01


class loss_cross_entropy_bounded(nn.Module):
    """
    Bounded Cross Entropy Loss from Perez-Ortiz et al., Tighter risk certificates for neural networks, 2020
    """

    def forward(self, outputs, targets, pmin=1e-4, bounded=True):
        """
        output.shape = (batch_size, num_classes) in [-inf, 0]
        """
        empirical_risk = F.nll_loss(outputs, targets)
        if bounded:
            empirical_risk = (1.0 / (np.log(1.0 / pmin))) * empirical_risk
        return empirical_risk

    def correct(self, output, targets):
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(targets.view_as(pred)).sum().item()
        return correct
