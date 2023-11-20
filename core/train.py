from typing import Dict
from functools import partial

# import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from . import bounds, loss as lloss

DEVICE = torch.device("cpu")


def mcsampling(
    data_loader: DataLoader,
    net: nn.Module,
    loss_fn: torch.nn,
    device: torch.device,
    mc_samples: int = 1,
) -> Dict:
    net.eval()
    net.to(device)
    loss_fn.to(device)
    mc_losses, mc_accuracy, mc_number = 0.0, 0.0, 0.0
    for batch_id, (data, labels) in enumerate(data_loader):
        data, labels = data.to(device), labels.to(device)
        labels = torch.reshape(labels, (-1, 1))  # TODO
        lossses_mc = 0.0
        for _ in range(mc_samples):
            output = net(data)
            loss_mc = loss_fn(output, labels)
            output_predictions = (output > 0).float() - (output <= 0).float()
            mc_accuracy += (output_predictions == labels).sum()
            mc_number += len(output_predictions)
            lossses_mc += loss_mc.detach().cpu()
        mc_losses += lossses_mc / mc_samples
    mc_losses /= batch_id + 1
    mc_accuracy /= mc_number
    return {"loss_mc": mc_losses, "accuracy_mc": mc_accuracy}


def trainNNet(
    net,
    optimizer_name,
    learning_rate,
    momentum,
    epochs,
    train_loader,
    loss_fn=lloss.loss_nll(),
    device=DEVICE,
    dryrun=False,
):
    """
    Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py#L1119

    Train function for a standard NN (including CNN)

    Parameters
    ----------
    net : NNet/CNNet object
        Network object to train

    optimizer : optim object
        Optimizer to use (e.g. SGD/Adam)

    epochs : int
        Number of training epochs

    train_loader: DataLoader object
        Train loader to use for training

    device : string
        Device the code will run in (e.g. 'cuda')

    """
    net.train()
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    else:
        raise ValueError
    for _ in range(epochs):
        correct, total, avgloss = 0.0, 0.0, 0.0
        for batch_id, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            net.zero_grad()
            outputs = net(data)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
            correct += loss_fn.correct(outputs, target)
            total += target.size(0)
            avgloss += loss.detach()
            if dryrun:
                break
        if dryrun:
            break

    metrics = {
        "loss": avgloss / (batch_id + 1),
        "count": (batch_id + 1),
        "num_examples": total,
        "accuracy": correct / total,
    }
    return net, metrics


def testNNet(net, test_loader, loss_fn=lloss.loss_nll(), device=DEVICE, dryrun=False):
    """
    Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py#L1119

    Test function for a standard NN (including CNN)

    Parameters
    ----------
    net : NNet/CNNet object
        Network object to train

    test_loader: DataLoader object
        Test data loader

    device : string
        Device the code will run in (e.g. 'cuda')

    """
    net.eval()
    correct, total, avgloss = 0, 0.0, 0.0
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            loss = loss_fn(outputs, target)
            correct += loss_fn.correct(outputs, target)
            total += target.size(0)
            avgloss += loss.detach()
            if dryrun:
                break
    metrics = {
        "loss": avgloss / (batch_id + 1),
        "count": (batch_id + 1),
        "num_examples": total,
        "accuracy": correct / total,
    }
    return metrics


def trainPNNet(
    net,
    optimizer_name,
    learning_rate,
    momentum,
    pbobj_cfg,
    epochs,
    train_loader,
    loss_fn=lloss.loss_cross_entropy_bounded(),
    device=DEVICE,
    lambda_var=None,
    learning_rate_lambda=None,
    momentum_lambda=None,
    dryrun=False,
):
    """
    Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py#L1119

    Train function for a probabilistic NN (including CNN)

    Parameters
    ----------
    net : ProbNNet/ProbCNNet object
        Network object to train

    optimizer : optim object
        Optimizer to use (e.g. SGD/Adam)

    pbobj : pbobj object
        PAC-Bayes inspired training objective to use for training

    epochs : int
        Number of training epochs

    train_loader: DataLoader object
        Train loader to use for training

    lambda_var : Lambda_var object
        Lambda variable for training objective flamb

    optimizer_lambda : optim object
        Optimizer to use for the learning the lambda_variable

    device : string
        Device the code will run in (e.g. 'cuda')

    """
    net.train()
    optim_partial = partial(torch.optim.SGD, lr=learning_rate, momentum=momentum)
    if pbobj_cfg.objective == "flamb":
        lambda_var.train()
        optimizer = optim_partial(
            [
                {"params": net.parameters()},
                {"params": [lambda_var], "lr": learning_rate_lambda, "momentum": momentum_lambda},
            ]
        )
    elif pbobj_cfg.objective == "bre":
        optimizer = torch.optim.RMSprop(net.parameters(), alpha=0.9, lr=learning_rate)
    else:
        optimizer = optim_partial(net.parameters())
    # variables that keep information about the results of optimising the bound
    for _ in range(epochs):
        avgerr, avgbound, avgkl, avgloss = 0.0, 0.0, 0.0, 0.0

        if pbobj_cfg.objective == "bbb":
            clamping = False
        else:
            clamping = True

        for batch_id, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            bound, kl_n, _, loss, err = bounds.train_obj(
                pbobj_cfg, net, data, target, loss_fn=loss_fn, lambda_var=lambda_var, clamping=clamping, dryrun=dryrun
            )
            bound.backward()
            optimizer.step()
            avgbound += bound.item()
            avgkl += kl_n
            avgloss += loss.item()
            avgerr += err
            if dryrun:
                break
        if dryrun:
            break
    metrics = {}
    metrics["train_obj"] = avgbound / (batch_id + 1)
    metrics["accuracy"] = 1 - avgerr / (batch_id + 1)
    metrics["KL_n"] = avgkl / (batch_id + 1)
    metrics["loss"] = avgloss / (batch_id + 1)
    metrics["num_examples"] = len(train_loader.dataset)
    return net, metrics


def testStochastic(
    net, test_loader, pbobj_cfg, loss_fn=lloss.loss_cross_entropy_bounded(), device=DEVICE, dryrun=False
):
    """
    Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py#L1119

    Test function for the stochastic predictor using a PNN

    Parameters
    ----------
    net : PNNet/PCNNet object
        Network object to test

    test_loader: DataLoader object
        Test data loader

    pbobj : pbobj object
        PAC-Bayes inspired training objective used during training

    device : string
        Device the code will run in (e.g. 'cuda')

    """
    # compute stochastic test accuracy
    net.eval()
    correct, loss_stchs, total = 0, 0.0, 0.0
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            batch_size = len(data)
            outputs = torch.zeros(batch_size, pbobj_cfg.classes).to(device)
            data, target = data.to(device), target.to(device)
            for i in range(batch_size):
                outputs[i, :] = net(data[i : i + 1], sample=True, clamping=True, pmin=pbobj_cfg.pmin)
                if dryrun:
                    break
            loss_stch = loss_fn(outputs, target, pmin=pbobj_cfg.pmin, bounded=True)
            correct += loss_fn.correct(outputs, target)
            total += batch_size
            loss_stchs += loss_stch
            if dryrun:
                break
    metrics = {
        "loss_stch": loss_stchs / (batch_id + 1),
        "count_stch": (batch_id + 1),
        "accuracy_stch": (correct / total),
        "num_examples_stch": total,
    }
    return metrics


def testPosteriorMean(
    net, test_loader, pbobj_cfg, loss_fn=lloss.loss_cross_entropy_bounded(), device=DEVICE, dryrun=False
):
    """
    Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py#L1119

    Test function for the deterministic predictor using a PNN
    (uses the posterior mean)

    Parameters
    ----------
    net : PNNet/PCNNet object
        Network object to test

    test_loader: DataLoader object
        Test data loader

    pbobj : pbobj object
        PAC-Bayes inspired training objective used during training

    device : string
        Device the code will run in (e.g. 'cuda')

    """
    net.eval()
    cross_entropies = 0.0
    correct, total = 0, 0.0
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            outputs = net(data, sample=False, clamping=True, pmin=pbobj_cfg.pmin)
            cross_entropy = loss_fn(outputs, target, pmin=pbobj_cfg.pmin, bounded=True)
            correct += loss_fn.correct(outputs, target)
            total += target.size(0)
            cross_entropies += cross_entropy
            if dryrun:
                break
    metrics = {
        "loss_mean": cross_entropies / (batch_id + 1),
        "count_mean": (batch_id + 1),
        "accuracy_mean": (correct / total),
        "num_examples_mean": total,
    }
    return metrics


def testEnsemble(
    net, test_loader, pbobj_cfg, loss_fn=lloss.loss_cross_entropy_bounded(), device=DEVICE, samples=100, dryrun=False
):
    """
    Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py#L1119

    Test function for the ensemble predictor using a PNN

    Parameters
    ----------
    net : PNNet/PCNNet object
        Network object to test

    test_loader: DataLoader object
        Test data loader

    pbobj : pbobj object
        PAC-Bayes inspired training objective used during training

    device : string
        Device the code will run in (e.g. 'cuda')

    samples: int
        Number of times to sample weights (i.e. members of the ensembles)

    """
    net.eval()
    correct, cross_entropies, total = 0, 0.0, 0.0
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            batch_size = len(data)
            outputs = torch.zeros([samples, batch_size, pbobj_cfg.classes]).to(device)
            for i in range(samples):
                outputs[i] = net(data, sample=True, clamping=True, pmin=pbobj_cfg.pmin)
                if dryrun:
                    break
            avgoutput = outputs.mean(0)
            cross_entropy = loss_fn(avgoutput, target, pmin=pbobj_cfg.pmin, bounded=True)
            correct += loss_fn.correct(avgoutput, target)
            total += batch_size
            cross_entropies += cross_entropy
            if dryrun:
                break
    metrics = {
        "loss_ens": cross_entropies / (batch_id + 1),
        "count_ens": (batch_id + 1),
        "accuracy_ens": (correct / total),
        "num_examples_ens": total,
    }
    return metrics
