# pylint: disable=too-many-arguments
"""Defines the MNIST Flower Client and a function to instantiate it."""
import copy

from collections import OrderedDict
from typing import Callable, Dict, Tuple, Optional

import flwr as fl
import torch

from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader
from omegaconf import open_dict

from . import loss, model, train, bounds

from .dataset import load_datasets


class FlowerClientGenFLBaseClass(fl.client.NumPyClient):
    """Standard Flower client for training classical (not probabilistic) NN."""

    def __init__(
        self,
        net: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: torch.nn.Module = loss.loss_logistic(),
    ):
        super().__init__()
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Changes the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
        self.set_parameters(parameters)

        # Set Hyperparameters
        optimizer_name = config["optimizer_name"]
        learning_rate = config["learning_rate"]
        momentum = config["momentum"]
        num_epochs = config["num_epochs"]
        device = config["device"]
        dryrun = config["dryrun"]

        # Train
        self.net.to(device)
        self.net, metrics = train.trainNNet(
            net=self.net,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            momentum=momentum,
            epochs=num_epochs,
            train_loader=self.train_loader,
            loss_fn=self.loss_fn,
            device=device,
            dryrun=dryrun,
        )
        return self.get_parameters({}), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)

        # Set Up
        device = config["device"]
        dryrun = config["dryrun"]

        # Set Hyperparameters
        metrics = train.testNNet(
            net=self.net,
            test_loader=self.val_loader,
            loss_fn=self.loss_fn,
            device=device,
            dryrun=dryrun,
        )

        return float(metrics["loss"]), len(self.val_loader.dataset), metrics


def gen_client_genfl_base_class_fn(
    net_cls: Callable[[], torch.nn.Module],
    device: torch.device,
    data_path: str,
    partition_type: str,
    perc_val: float,
    perc_data: float,
    num_clients: int,
    batch_size_client: int,
    batch_size_server: int,
    seed: Optional[int] = 42,
) -> Tuple[Callable[[str], FlowerClientGenFLBaseClass], DataLoader]:
    train_loaders, _, val_loaders, _, test_loader = load_datasets(
        data_path=data_path,
        num_clients=num_clients,
        partition_type=partition_type,
        perc_val=perc_val,
        batch_size_client=batch_size_client,
        batch_size_server=batch_size_server,
        logistic=True,
        seed=seed,
        perc_data=perc_data,
    )

    client_global_sizes = {
        "train": sum(len(train_loader.dataset) for train_loader in train_loaders),
        "val": sum(len(val_loader.dataset) for val_loader in val_loaders),
    }

    def client_fn(cid: str) -> FlowerClientGenFLBaseClass:
        """Create a Flower client representing a single organization."""

        # Load model
        net = net_cls().to(device)

        # Note: each client gets a different train_loader/val_loader, so each client
        # will train and evaluate on their own unique data
        train_loader = train_loaders[int(cid)]
        val_loader = val_loaders[int(cid)]

        # Create a  single Flower client representing a single organization
        return FlowerClientGenFLBaseClass(net, train_loader, val_loader)

    return client_fn, test_loader, client_global_sizes


class FlowerClientGenFLPrior(FlowerClientGenFLBaseClass):
    def __init__(
        self,
        net: torch.nn.Module,
        prior_loader: DataLoader,
        loss_fn: torch.nn.Module,
    ):
        super().__init__(net, prior_loader, prior_loader, loss_fn)  # train and val loaders become prior_loader


def gen_client_genfl_prior_fn(
    data_path: str,
    partition_type: str,
    perc_val: float,
    perc_test: float,
    perc_data: float,
    net_cls: Callable[[], torch.nn.Module],
    dropout_prob: float,
    loss_fn: Callable,
    batch_size_client: int,
    batch_size_server: int,
    num_clients: int,
    seed: Optional[int] = 42,
    logistic=False,
) -> Tuple[Callable[[str], FlowerClientGenFLPrior], DataLoader, Dict]:
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
    device : torch.device
        The device on which the the client will train on and test on.
    iid : bool
        The way to partition the data for each client, i.e. whether the data
        should be independent and identically distributed between the clients
        or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario)
    balance : bool
        Whether the dataset should contain an equal number of samples in each class,
        by default True
    num_clients : int
        The number of clients present in the setup
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    batch_size : int
        The size of the local batches each client trains on.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """

    # In this context val_loaders/prior_loaders will be used for training a prior
    train_loaders, _, prior_loaders, _, test_loader = load_datasets(
        data_path=data_path,
        num_clients=num_clients,
        partition_type=partition_type,
        perc_val=perc_val,
        perc_test=perc_test,
        batch_size_client=batch_size_client,
        batch_size_server=batch_size_server,
        logistic=logistic,
        seed=seed,
        perc_data=perc_data,
    )

    client_global_sizes = {
        "train": sum(len(train_loader.dataset) for train_loader in train_loaders),
        "prior": sum(len(prior_loader.dataset) for prior_loader in prior_loaders),
    }

    def client_fn(cid: str) -> FlowerClientGenFLPrior:
        """Create a Flower client representing a single organization."""

        # Load model
        net = net_cls(dropout_prob=dropout_prob)  # dropout required because not saved in state_dict

        # Note: each client gets a different train_loader/val_loader, so each client
        # will train and evaluate on their own unique data
        train_loader = train_loaders[int(cid)]
        prior_loader = prior_loaders[int(cid)]

        # Create a  single Flower client representing a single organization
        return FlowerClientGenFLPrior(
            net=net,
            prior_loader=train_loader if logistic else prior_loader,  # distinction between Perez and Dziugaite
            loss_fn=loss_fn,
        )

    return client_fn, test_loader, client_global_sizes


class FlowerClientGenFLPosterior(FlowerClientGenFLBaseClass):
    def __init__(
        self,
        net: torch.nn.Module,
        train_loader: DataLoader,
        whole_train: DataLoader,
        loss_fn: torch.nn.Module,
    ):
        super().__init__(net, train_loader, None, loss_fn)
        self.whole_train = whole_train

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        self.set_parameters(parameters)

        # Set Hyperparameters
        optimizer_name = config["optimizer_name"]
        learning_rate = config["learning_rate"]
        momentum = config["momentum"]
        num_epochs = config["num_epochs"]
        device = config["device"]
        dryrun = config["dryrun"]
        if "lambda_var" in config:
            lambda_var = model.Lambda_var(lamb=config["lambda_var"], n=config["train_size"])
            learning_rate_lambda = config["learning_rate_lambda"]
            momentum_lambda = config["momentum_lambda"]
        else:
            lambda_var = None
            learning_rate_lambda = None
            momentum_lambda = None

        pbobj_cfg = config["pbobj_cfg"]
        if config["is_local_pbobj_cfg"]:  # for personalized steps, optimize bound on local dataset sizes
            pbobj_cfg = copy.deepcopy(pbobj_cfg)
            with open_dict(pbobj_cfg):
                pbobj_cfg.n_posterior = len(self.train_loader.dataset)
                pbobj_cfg.n_bound = len(self.train_loader.dataset)

        # Train
        self.net, metrics = train.trainPNNet(
            net=self.net,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            momentum=momentum,
            pbobj_cfg=pbobj_cfg,
            epochs=num_epochs,
            train_loader=self.train_loader,
            loss_fn=self.loss_fn,
            lambda_var=lambda_var,
            learning_rate_lambda=learning_rate_lambda,
            momentum_lambda=momentum_lambda,
            device=device,
            dryrun=dryrun,
        )
        if "lambda_var" in config:
            metrics["lambda_var"] = lambda_var.lamb.item()
        return self.get_parameters({}), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)

        # Set Up
        pbobj_cfg = config["pbobj_cfg"]
        device = config["device"]
        dryrun = config["dryrun"]

        # Eval
        metrics_mean = train.testPosteriorMean(
            net=self.net,
            test_loader=self.train_loader,
            pbobj_cfg=pbobj_cfg,
            loss_fn=self.loss_fn,
            device=device,
            dryrun=dryrun,
        )

        if dryrun or config["compute_risk_certificates"] or config["compute_client_risk_certificates"]:
            metrics_rc = bounds.pre_compute_risk_certificates_client(
                net=self.net,
                toolarge=config["toolarge"],
                pbobj_cfg=pbobj_cfg,
                loss_fn=self.loss_fn,
                device=device,
                train_loader=self.train_loader,
                whole_train=self.whole_train,
                dryrun=dryrun,
            )
        else:
            metrics_rc = {}
        if dryrun or config["compute_client_risk_certificates"]:
            kl = self.net.compute_kl()
            second_cfg = copy.deepcopy(pbobj_cfg)
            with open_dict(second_cfg):
                second_cfg.n_posterior = len(self.train_loader.dataset)
                second_cfg.n_bound = len(self.train_loader.dataset)
            (
                train_obj,
                kl_n,
                empirical_risk_ce,
                empirical_risk_01,
                risk_ce,
                risk_01,
            ) = bounds.compute_final_stats_risk_server(
                error_ce=metrics_rc["error_ce"],
                error_01=metrics_rc["error_01"],
                cfg=second_cfg,
                kl=kl,
                lambda_var=config.get("lambda_var", None),
                lambda_disc=getattr(self.net, "sigma_prior", None),
                dryrun=config["dryrun"],
            )
            metrics_client = {}
            metrics_client["train_obj_c"] = train_obj
            metrics_client["kl_n_c"] = kl_n
            metrics_client["empirical_risk_ce_c"] = empirical_risk_ce
            metrics_client["empirical_risk_01_c"] = empirical_risk_01
            metrics_client["risk_ce_c"] = risk_ce
            metrics_client["risk_01_c"] = risk_01
        else:
            metrics_client = {}
        metrics = metrics_mean | metrics_rc | metrics_client

        return float(metrics["loss_mean"]), len(self.train_loader.dataset), metrics


def gen_client_genfl_posterior_fn(
    # Data parameters
    data_path: str,
    partition_type: str,
    perc_data: float,
    perc_val: float,
    perc_test: float,
    # Module parameters
    net_cls: Callable[[], torch.nn.Module],
    rho_prior: float,
    init_net: torch.nn.Module,
    loss_fn: Callable,
    # Optim parameters
    batch_size_client: int,
    batch_size_server: int,
    # Federated Learning parameters
    num_clients: int,
    # Misc
    device: torch.device,
    seed: Optional[int] = 42,
    logistic=False,
) -> Tuple[Callable[[str], FlowerClientGenFLPosterior], DataLoader, Dict]:
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
    device : torch.device
        The device on which the the client will train on and test on.
    iid : bool
        The way to partition the data for each client, i.e. whether the data
        should be independent and identically distributed between the clients
        or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario)
    balance : bool
        Whether the dataset should contain an equal number of samples in each class,
        by default True
    num_clients : int
        The number of clients present in the setup
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    batch_size : int
        The size of the local batches each client trains on.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """

    # In this context prior_loaders will be used for training a prior
    train_loaders, whole_trains, _, _, test_loader = load_datasets(
        data_path=data_path,
        num_clients=num_clients,
        partition_type=partition_type,
        perc_val=perc_val,
        perc_test=perc_test,
        batch_size_client=batch_size_client,
        batch_size_server=batch_size_server,
        logistic=logistic,
        seed=seed,
        perc_data=perc_data,
    )

    client_global_sizes = {
        "train": sum(len(train_loader.dataset) for train_loader in train_loaders),
    }

    def client_fn(cid: str) -> FlowerClientGenFLPosterior:
        """Create a Flower client representing a single organization."""

        # Load model
        net = net_cls(rho_prior=rho_prior, init_net=init_net).to(device)

        # Get respective dataloaders
        train_loader = train_loaders[int(cid)]
        whole_train = whole_trains[int(cid)]

        # Return a FL client
        return FlowerClientGenFLPosterior(
            net=net,
            train_loader=train_loader,
            whole_train=whole_train,
            loss_fn=loss_fn,
        )

    return client_fn, test_loader, client_global_sizes


class FlowerClientGenFLPersonalized(FlowerClientGenFLBaseClass):
    def __init__(
        self,
        net: torch.nn.Module,
        train_loader: DataLoader,
        prior_loader: DataLoader,
        test_loader: DataLoader,
        whole_train: DataLoader,
        loss_fn: torch.nn.Module,
    ):
        super().__init__(net, train_loader, None, loss_fn)
        self.prior_loader = prior_loader
        self.test_loader = test_loader
        self.whole_train = whole_train

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
        # Set parameters
        self.set_parameters(parameters)

        # Set Hyperparameters
        optimizer_name = config["optimizer_name"]
        learning_rate = config["learning_rate"]
        momentum = config["momentum"]
        num_epochs = config["num_epochs"]
        device = config["device"]
        dryrun = config["dryrun"]
        if "lambda_var" in config:
            lambda_var = model.Lambda_var(lamb=config["lambda_var"], n=config["train_size"])
            learning_rate_lambda = config["learning_rate_lambda"]
            momentum_lambda = config["momentum_lambda"]
        else:
            lambda_var = None
            learning_rate_lambda = None
            momentum_lambda = None

        # Second config for local bound optimization in personalized step
        pbobj_cfg = config["pbobj_cfg"]
        second_pbobj_cfg = copy.deepcopy(pbobj_cfg)
        with open_dict(pbobj_cfg):
            second_pbobj_cfg.n_posterior = len(self.train_loader.dataset)
            second_pbobj_cfg.n_bound = len(self.train_loader.dataset)

        # Train
        self.net.train()
        self.net, metrics = train.trainPNNet(
            net=self.net,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            momentum=momentum,
            pbobj_cfg=second_pbobj_cfg,
            epochs=num_epochs,
            train_loader=self.train_loader,
            loss_fn=self.loss_fn,
            lambda_var=lambda_var,
            learning_rate_lambda=learning_rate_lambda,
            momentum_lambda=momentum_lambda,
            device=device,
            dryrun=dryrun,
        )
        if "lambda_var" in config:
            metrics["lambda_var"] = lambda_var.lamb.item()

        # Eval on trained personalized parameters
        self.net.eval()
        metrics_mean = train.testPosteriorMean(
            net=self.net,
            test_loader=self.test_loader,
            pbobj_cfg=second_pbobj_cfg,
            loss_fn=self.loss_fn,
            device=device,
            dryrun=dryrun,
        )
        metrics_stch = train.testStochastic(
            net=self.net,
            test_loader=self.test_loader,
            pbobj_cfg=second_pbobj_cfg,
            loss_fn=self.loss_fn,
            device=device,
            dryrun=dryrun,
        )
        metrics_ens = train.testEnsemble(
            net=self.net,
            test_loader=self.test_loader,
            pbobj_cfg=second_pbobj_cfg,
            loss_fn=self.loss_fn,
            device=device,
            dryrun=dryrun,
        )
        metrics_rc = bounds.pre_compute_risk_certificates_client(
            net=self.net,
            toolarge=config["toolarge"],
            pbobj_cfg=second_pbobj_cfg,
            loss_fn=self.loss_fn,
            device=device,
            train_loader=self.train_loader,
            whole_train=self.whole_train,
            dryrun=dryrun,
        )
        kl = self.net.compute_kl()

        (
            train_obj,
            kl_n,
            empirical_risk_ce,
            empirical_risk_01,
            risk_ce,
            risk_01,
        ) = bounds.compute_final_stats_risk_server(
            error_ce=metrics_rc["error_ce"],
            error_01=metrics_rc["error_01"],
            cfg=second_pbobj_cfg,
            kl=kl,
            lambda_var=config.get("lambda_var", None),
            lambda_disc=getattr(self.net, "sigma_prior", None),
            dryrun=config["dryrun"],
        )
        metrics_client = {}
        metrics_client["train_obj_c"] = train_obj
        metrics_client["kl_n_c"] = kl_n
        metrics_client["empirical_risk_ce_c"] = empirical_risk_ce
        metrics_client["empirical_risk_01_c"] = empirical_risk_01
        metrics_client["risk_ce_c"] = risk_ce
        metrics_client["risk_01_c"] = risk_01

        metrics = metrics | metrics_mean | metrics_stch | metrics_ens | metrics_rc | metrics_client
        return self.get_parameters({}), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        """Should return nothing meaningful because the client is not supposed to evaluate from server's parameters"""
        return 1.0, 1, {}


def gen_client_genfl_personalized_fn(
    # Data parameters
    data_path: str,
    partition_type: str,
    perc_data: float,
    perc_val: float,
    perc_test: float,
    # Module parameters
    net_cls: Callable[[], torch.nn.Module],
    rho_prior: float,
    loss_fn: Callable,
    # Optim parameters
    batch_size_client: int,
    batch_size_server: int,
    # Federated Learning parameters
    num_clients: int,
    # Misc
    seed: Optional[int] = 42,
) -> Tuple[Callable[[str], FlowerClientGenFLPersonalized], DataLoader, Dict]:
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
    device : torch.device
        The device on which the the client will train on and test on.
    iid : bool
        The way to partition the data for each client, i.e. whether the data
        should be independent and identically distributed between the clients
        or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario)
    balance : bool
        Whether the dataset should contain an equal number of samples in each class,
        by default True
    num_clients : int
        The number of clients present in the setup
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    batch_size : int
        The size of the local batches each client trains on.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """

    # In this context prior_loaders will be used for training a prior
    train_loaders, whole_trains, prior_loaders, test_loaders, test_loader = load_datasets(
        data_path=data_path,
        num_clients=num_clients,
        partition_type=partition_type,
        perc_val=perc_val,
        perc_test=perc_test,
        batch_size_client=batch_size_client,
        batch_size_server=batch_size_server,
        logistic=False,  # Not done with dziugaite
        seed=seed,
        perc_data=perc_data,
    )

    client_global_sizes = {
        "train": sum(len(train_loader.dataset) for train_loader in train_loaders),
        "prior": sum(len(prior_loader.dataset) for prior_loader in prior_loaders),
        "test": sum(len(_test_loader.dataset) for _test_loader in test_loaders),
    }

    def client_fn(cid: str) -> FlowerClientGenFLPersonalized:
        """Create a Flower client representing a single organization."""

        # Load model
        net = net_cls(rho_prior=rho_prior)

        # Get respective dataloaders
        train_loader = train_loaders[int(cid)]
        prior_loader = prior_loaders[int(cid)]
        whole_train = whole_trains[int(cid)]
        test_loader = test_loaders[int(cid)]

        # Return a FL client
        return FlowerClientGenFLPersonalized(
            net=net,
            train_loader=train_loader,
            prior_loader=prior_loader,
            test_loader=test_loader,
            whole_train=whole_train,
            loss_fn=loss_fn,
        )

    return client_fn, test_loader, client_global_sizes
