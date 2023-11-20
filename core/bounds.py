"""
Code from the github of the paper from Perez-Ortiz et al., Tighter risk certificates for neural networks, 2020

Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/bounds.py
"""
import math
import numpy as np

import torch

DEVICE = torch.device("cpu")


def _compute_losses(cfg, net, data, target, loss_fn, clamping=True, dryrun=False):
    # compute both cross entropy and 01 loss
    # returns outputs of the network as well
    outputs = net(data, sample=True, clamping=clamping, pmin=cfg.pmin)
    loss = loss_fn(outputs, target, pmin=cfg.pmin, bounded=clamping)
    correct = loss_fn.correct(outputs, target)
    total = target.size(0)
    loss_01 = 1 - (correct / total)
    return loss, loss_01, outputs


def bound(cfg, empirical_risk, kl, train_size, lambda_var=None, lamba_disc=None, dryrun=False):
    # compute training objectives
    if cfg.objective == "fquad":
        kl = kl * cfg.kl_penalty
        repeated_kl_ratio = torch.div(kl + np.log((2 * np.sqrt(train_size)) / cfg.delta), 2 * train_size)
        first_term = torch.sqrt(empirical_risk + repeated_kl_ratio)
        second_term = torch.sqrt(repeated_kl_ratio)
        train_obj = torch.pow(first_term + second_term, 2)
    elif cfg.objective == "flamb":
        kl = kl * cfg.kl_penalty
        lamb = lambda_var.lamb_scaled
        kl_term = torch.div(kl + np.log((2 * np.sqrt(train_size)) / cfg.delta), train_size * lamb * (1 - lamb / 2))
        first_term = torch.div(empirical_risk, 1 - lamb / 2)
        train_obj = first_term + kl_term
    elif cfg.objective == "fclassic":
        kl = kl * cfg.kl_penalty
        kl_ratio = torch.div(kl + np.log((2 * np.sqrt(train_size)) / cfg.delta), 2 * train_size)
        train_obj = empirical_risk + torch.sqrt(kl_ratio)
    elif cfg.objective == "bbb":
        # ipdb.set_trace()
        train_obj = empirical_risk + cfg.kl_penalty * (kl / train_size)
    elif cfg.objective == "bre":  # B_{RE} objective from Dziugaite et al., "Computing Nonvacuous ...", 2017
        kl = kl * cfg.kl_penalty
        # Other terms
        factor1 = 2 * torch.log(torch.tensor(cfg.log_prior_std_precision))
        factor2 = 2 * torch.log(
            torch.maximum((torch.log(torch.tensor(cfg.log_prior_std_base)) - torch.log(lamba_disc)), torch.tensor(1e-2))
        )  # the max with 1e-2 is a trick from dziugaite to avoid numerical issues
        Oth_term = factor1 + factor2 + torch.log(torch.tensor(math.pi**2 * train_size / (6 * cfg.delta)))
        # train objective
        Bquad = kl + Oth_term
        train_obj = empirical_risk + torch.sqrt((Bquad / (2 * (train_size - 1))))
    else:
        raise RuntimeError(f"Wrong objective {cfg.objective}")
    return train_obj


def _mcsampling(
    cfg, net, input, target, loss_fn, batches=True, clamping=True, data_loader=None, device=DEVICE, dryrun=False
):
    # compute empirical risk with Monte Carlo sampling
    error = 0.0
    cross_entropy = 0.0
    if batches:
        for batch_id, (data_batch, target_batch) in enumerate(data_loader):
            data_batch, target_batch = data_batch.to(device), target_batch.to(device)
            cross_entropy_mc = 0.0
            error_mc = 0.0
            for _ in range(cfg.mc_samples):
                loss, loss_01, _ = _compute_losses(
                    cfg=cfg,
                    net=net,
                    data=data_batch,
                    target=target_batch,
                    loss_fn=loss_fn,
                    clamping=clamping,
                )
                cross_entropy_mc += loss
                error_mc += loss_01
                if dryrun:
                    break

            # we average cross-entropy and 0-1 error over all MC samples
            cross_entropy += cross_entropy_mc / cfg.mc_samples
            error += error_mc / cfg.mc_samples
            if dryrun:
                break
        # we average cross-entropy and 0-1 error over all batches
        cross_entropy /= batch_id + 1
        error /= batch_id + 1
    else:
        cross_entropy_mc = 0.0
        error_mc = 0.0
        for _ in range(cfg.mc_samples):
            loss, loss_01, _ = _compute_losses(
                cfg=cfg, net=net, data=input, target=target, loss_fn=loss_fn, clamping=clamping
            )
            cross_entropy_mc += loss
            error_mc += loss_01
            if dryrun:
                break
            # we average cross-entropy and 0-1 error over all MC samples
        cross_entropy += cross_entropy_mc / cfg.mc_samples
        error += error_mc / cfg.mc_samples
    return cross_entropy, error


def train_obj(cfg, net, input, target, loss_fn, clamping=True, lambda_var=None, dryrun=False):
    # compute train objective and return all metrics
    kl = net.compute_kl()
    loss, loss_01, outputs = _compute_losses(
        cfg=cfg, net=net, data=input, target=target, loss_fn=loss_fn, clamping=clamping
    )
    if hasattr(net, "sigma_prior"):
        lambda_disc = torch.pow(getattr(net, "sigma_prior", None), 2)  # sigma^2 = lambda (the variance)
    else:
        lambda_disc = None
    train_obj = bound(
        cfg=cfg,
        empirical_risk=loss,
        kl=kl,
        train_size=cfg.n_posterior,
        lambda_var=lambda_var,
        lamba_disc=lambda_disc,
    )
    return train_obj, kl / cfg.n_posterior, outputs, loss, loss_01


def compute_risk_certificates(
    net,
    toolarge,
    pbobj_cfg,
    loss_fn,
    device=DEVICE,
    lambda_var=None,
    train_loader=None,
    whole_train=None,
    dryrun=False,
):
    """
    Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py#L1119

    Function to compute risk certificates and other statistics at the end of training

    Parameters
    ----------
    net : PNNet/PCNNet object
        Network object to test

    toolarge: bool
        Whether the dataset is too large to fit in memory (computation done in batches otherwise)

    pbobj : pbobj object
        PAC-Bayes inspired training objective used during training

    device : string
        Device the code will run in (e.g. 'cuda')

    lambda_var : Lambda_var object
        Lambda variable for training objective flamb

    train_loader: DataLoader object
        Data loader for computing the risk certificate (multiple batches, used if toolarge=True)

    whole_train: DataLoader object
        Data loader for computing the risk certificate (one unique batch, used if toolarge=False)

    """
    net.eval()
    with torch.no_grad():
        if toolarge:
            train_obj, kl, empirical_risk_ce, empirical_risk_01, risk_ce, risk_01 = compute_final_stats_risk(
                pbobj_cfg,
                net,
                loss_fn=loss_fn,
                lambda_var=lambda_var,
                lambda_disc=getattr(net, "sigma_prior", None),
                clamping=True,
                data_loader=train_loader,
                device=device,
                dryrun=dryrun,
            )
        else:
            # a bit hacky, we load the whole dataset to compute the bound
            for data, target in whole_train:
                data, target = data.to(device), target.to(device)
                train_obj, kl, empirical_risk_ce, empirical_risk_01, risk_ce, risk_01 = compute_final_stats_risk(
                    pbobj_cfg,
                    net,
                    loss_fn=loss_fn,
                    lambda_var=lambda_var,
                    lambda_disc=getattr(net, "sigma_prior", None),
                    clamping=True,
                    input=data,
                    target=target,
                    device=device,
                    dryrun=dryrun,
                )
    metrics = {
        "train_obj": train_obj,
        "risk_ce": risk_ce,
        "risk_01": risk_01,
        "kl": kl,
        "empirical_risk_ce": empirical_risk_ce,
        "empirical_risk_01": empirical_risk_01,
    }
    return metrics


def pre_compute_risk_certificates_client(
    net, toolarge, pbobj_cfg, loss_fn, device=DEVICE, train_loader=None, whole_train=None, dryrun=False
):
    """
    Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py#L1119

    Function to compute risk certificates and other statistics at the end of training

    Parameters
    ----------
    net : PNNet/PCNNet object
        Network object to test

    toolarge: bool
        Whether the dataset is too large to fit in memory (computation done in batches otherwise)

    pbobj : pbobj object
        PAC-Bayes inspired training objective used during training

    device : string
        Device the code will run in (e.g. 'cuda')

    lambda_var : Lambda_var object
        Lambda variable for training objective flamb

    train_loader: DataLoader object
        Data loader for computing the risk certificate (multiple batches, used if toolarge=True)

    whole_train: DataLoader object
        Data loader for computing the risk certificate (one unique batch, used if toolarge=False)

    """
    net.eval()
    with torch.no_grad():
        if toolarge:
            error_ce, error_01 = pre_compute_final_stats_risk_client(
                pbobj_cfg, net, loss_fn=loss_fn, clamping=True, data_loader=train_loader, dryrun=dryrun
            )
        else:
            # a bit hacky, we load the whole dataset to compute the bound
            for data, target in whole_train:
                data, target = data.to(device), target.to(device)
                error_ce, error_01 = pre_compute_final_stats_risk_client(
                    pbobj_cfg, net, loss_fn=loss_fn, clamping=True, input=data, target=target, dryrun=dryrun
                )
    metrics = {
        "error_ce": error_ce,
        "error_01": error_01,
    }
    return metrics


def compute_final_stats_risk(
    cfg,
    net,
    loss_fn,
    input=None,
    target=None,
    data_loader=None,
    clamping=True,
    device=DEVICE,
    lambda_var=None,
    dryrun=False,
):
    # compute all final stats and risk certificates
    kl = net.compute_kl()
    if data_loader:
        error_ce, error_01 = _mcsampling(
            cfg=cfg,
            net=net,
            input=input,
            target=target,
            loss_fn=loss_fn,
            batches=True,
            clamping=clamping,
            data_loader=data_loader,
            dryrun=dryrun,
            device=device,
        )
    else:
        error_ce, error_01 = _mcsampling(
            cfg=cfg,
            net=net,
            input=input,
            target=target,
            loss_fn=loss_fn,
            batches=False,
            clamping=clamping,
            dryrun=dryrun,
            device=device,
        )

    empirical_risk_ce = inv_kl(error_ce.item(), np.log(2 / cfg.delta_test) / cfg.mc_samples)
    empirical_risk_01 = inv_kl(error_01, np.log(2 / cfg.delta_test) / cfg.mc_samples)

    train_obj = bound(
        cfg=cfg,
        empirical_risk=empirical_risk_ce,
        kl=kl,
        train_size=cfg.n_posterior,
        lambda_var=lambda_var,
        lamba_disc=getattr(net, "sigma_prior", None),
        dryrun=dryrun,
    )

    risk_ce = inv_kl(empirical_risk_ce, (kl + np.log((2 * np.sqrt(cfg.n_bound)) / cfg.delta_test)) / cfg.n_bound)
    risk_01 = inv_kl(empirical_risk_01, (kl + np.log((2 * np.sqrt(cfg.n_bound)) / cfg.delta_test)) / cfg.n_bound)
    return train_obj.item(), kl.item() / cfg.n_bound, empirical_risk_ce, empirical_risk_01, risk_ce, risk_01


def pre_compute_final_stats_risk_client(
    cfg, net, loss_fn, input=None, target=None, data_loader=None, clamping=True, dryrun=False
):
    if data_loader:
        error_ce, error_01 = _mcsampling(
            cfg=cfg,
            net=net,
            input=input,
            loss_fn=loss_fn,
            target=target,
            batches=True,
            clamping=clamping,
            data_loader=data_loader,
            dryrun=dryrun,
        )
    else:
        error_ce, error_01 = _mcsampling(
            cfg=cfg,
            net=net,
            input=input,
            target=target,
            loss_fn=loss_fn,
            batches=False,
            clamping=clamping,
            dryrun=dryrun,
        )

    return error_ce, error_01


def compute_final_stats_risk_server(error_ce, error_01, cfg, kl, lambda_var=None, lambda_disc=None, dryrun=False):
    empirical_risk_ce = inv_kl(error_ce, np.log(2 / cfg.delta_test) / cfg.mc_samples)
    empirical_risk_01 = inv_kl(error_01, np.log(2 / cfg.delta_test) / cfg.mc_samples)

    train_obj = bound(
        cfg=cfg,
        empirical_risk=empirical_risk_ce,
        kl=kl,
        train_size=cfg.n_posterior,
        lambda_var=lambda_var,
        lamba_disc=lambda_disc,
        dryrun=dryrun,
    )

    risk_ce = inv_kl(empirical_risk_ce, (kl + np.log((2 * np.sqrt(cfg.n_bound)) / cfg.delta)) / cfg.n_bound)
    risk_01 = inv_kl(empirical_risk_01, (kl + np.log((2 * np.sqrt(cfg.n_bound)) / cfg.delta)) / cfg.n_bound)
    return train_obj.item(), kl.item() / cfg.n_bound, empirical_risk_ce, empirical_risk_01, risk_ce, risk_01


# ## Utils for computing PAC-Bayes Bounds ## #
def inv_kl(qs, ks):
    """
    Inversion of the binary kl

    Parameters
    ----------
    qs : float
        Empirical risk

    ks : float
        second term for the binary kl inversion

    """
    qd = 0
    ikl = 0
    izq = qs
    dch = 1 - 1e-10
    while (dch - izq) / dch >= 1e-5:
        p = (izq + dch) * 0.5
        if qs == 0:
            ikl = ks - (0 + (1 - qs) * math.log((1 - qs) / (1 - p)))
        elif qs == 1:
            ikl = ks - (qs * math.log(qs / p) + 0)
        else:
            ikl = ks - (qs * math.log(qs / p) + (1 - qs) * math.log((1 - qs) / (1 - p)))
        if ikl < 0:
            dch = p
        else:
            izq = p
        qd = p
    return qd


def approximate_BPAC_bound(train_accur, B_init, niter=5):
    B_RE = 2 * B_init**2
    A = 1 - train_accur
    B_next = B_init + A
    if B_next > 1.0:
        return 1.0
    for _ in range(niter):
        B_next = _Newt(B_next, A, B_RE)
    return B_next


def _Newt(p, q, c):
    newp = p - (_KLdiv(q, p) - c) / _KLdiv_prime(q, p)
    return newp


def _KLdiv(pbar, p):
    return pbar * np.log(pbar / p) + (1 - pbar) * np.log((1 - pbar) / (1 - p))


def _KLdiv_prime(pbar, p):
    return (1 - pbar) / (1 - p) - pbar / p


def discretize_log_lambda(log_lambda, log_prior_std_base=0.1, log_prior_std_precision=100):
    lambda_opt = torch.t_copy(torch.exp(2.0 * log_lambda)).detach().cpu().item()
    jopt = log_prior_std_precision * math.log(log_prior_std_base / lambda_opt)
    jdisc_up = np.float32(math.ceil(jopt))
    jdisc_down = np.float32(math.floor(jopt))
    init_log_prior_std_up = (np.log(log_prior_std_base) - jdisc_up / log_prior_std_precision) / 2
    init_log_prior_std_down = (np.log(log_prior_std_base) - jdisc_down / log_prior_std_precision) / 2
    return (init_log_prior_std_down, init_log_prior_std_up)


def compute_pac_bayes_bound(res_dict_down, res_dict_up):
    B_valD = np.array([x.cpu() for x in res_dict_down["B"]]).mean()
    B_valU = np.array([x.cpu() for x in res_dict_up["B"]]).mean()
    B_val = np.minimum(B_valD, B_valU)
    if B_valU < B_valD:
        mean_train_accuracy = res_dict_up["val_accuracy"][-1].cpu().item() / res_dict_up["val_number"][-1]
        is_down = False
    else:
        mean_train_accuracy = res_dict_down["val_accuracy"][-1].cpu().item() / res_dict_down["val_number"][-1]
        is_down = True

    return approximate_BPAC_bound(mean_train_accuracy, B_val), is_down
