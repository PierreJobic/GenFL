import hydra

from omegaconf import OmegaConf, DictConfig

from core import utils


@hydra.main(config_path="conf", config_name="genfl_prior")
def runexp(cfg: DictConfig):
    import os

    from pathlib import Path
    from logging import INFO
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd

    # from omegaconf import open_dict

    import torch
    import numpy as np
    import flwr as fl

    from flwr.server.client_manager import SimpleClientManager
    from flwr.common.logger import log

    from core import model, client, strategy as strat, server as svr, metrics, loss

    DEVICE = torch.device("cpu")

    OmegaConf.resolve(cfg)
    assert cfg.prior_type == "learnt" and cfg.perc_val != 0.0

    log(INFO, "\n" + OmegaConf.to_yaml(cfg))
    log(INFO, HydraConfig.get().job.override_dirname)

    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    eval('setattr(torch.backends.cudnn, "deterministic", True)')
    eval('setattr(torch.backends.cudnn, "benchmark", False)')

    # Load model
    if cfg.model_name == "NNet4lPerez":
        net_cls = model.NNet4lPerez
        loss_fn = loss.loss_nll()
    elif cfg.model_name == "NNet4lDziugaite":
        net_cls = model.NNet4lDziugaite
        loss_fn = loss.loss_logistic()
    net = net_cls(dropout_prob=cfg.dropout_prob).to(DEVICE)
    checkpoint_parameters = fl.common.ndarrays_to_parameters([val.cpu().numpy() for _, val in net.state_dict().items()])

    client_fn, testloader, client_global_sizes = client.gen_client_genfl_prior_fn(
        # Data parameters
        data_path=cfg.data_path,
        partition_type=cfg.partition_type,
        perc_data=cfg.perc_data,
        perc_val=cfg.perc_val,  # percentage of prior set in this case because validation set is used as train set
        perc_test=cfg.perc_test,
        # Module parameters
        net_cls=net_cls,
        dropout_prob=cfg.dropout_prob,
        loss_fn=loss_fn,
        # Optim parameters
        batch_size_client=cfg.batch_size_client,
        batch_size_server=cfg.batch_size_server,
        # Federated Learning parameters
        num_clients=cfg.num_clients,
        # Misc
        seed=seed,
        logistic=cfg.logistic,
    )

    # train_size = client_global_sizes["train"]
    # prior_size = client_global_sizes["prior"]
    # with open_dict(cfg):
    #     cfg.pbobj.n_posterior = prior_size
    #     cfg.pbobj.n_bound = prior_size
    #     log(
    #         INFO,
    #         f"Computed values: n_posterior: {cfg.pbobj.n_posterior} | "
    #         f"n_bound: {cfg.pbobj.n_bound} | nb classes: {cfg.pbobj.classes}",
    #     )

    evaluate_fn = strat.GenFLPrior.get_evaluate_fn(
        net=net,
        test_loader=testloader,
        loss_fn=loss_fn,
        device=DEVICE,
        dryrun=cfg.dryrun,
    )

    config_fn = strat.get_universal_config_fn(
        optimizer_name=cfg.optimizer_name_prior,
        learning_rate=cfg.learning_rate_prior,
        momentum=cfg.momentum_prior,
        num_epochs=cfg.num_epochs_prior,
        device=DEVICE,
        dryrun=cfg.dryrun,
    )

    strategy = strat.GenFLPrior(
        # FedAvg Parameters
        initial_parameters=checkpoint_parameters,
        min_available_clients=cfg.num_clients,
        accept_failures=cfg.get("accept_failures", False),
        # # Train
        min_fit_clients=max(1, int(cfg.num_clients * cfg.client_fraction)),
        fraction_fit=cfg.client_fraction,
        on_fit_config_fn=config_fn,
        fit_metrics_aggregation_fn=metrics.metrics_aggregation_universal_fn,
        # # Evaluate
        min_evaluate_clients=1,
        fraction_evaluate=cfg.fraction_evaluate,
        on_evaluate_config_fn=config_fn,
        evaluate_metrics_aggregation_fn=metrics.metrics_aggregation_universal_fn,
        evaluate_fn=evaluate_fn,
        # GenFL Parameters
        net=net,
        # Monitoring Parameters
        save_every_n_rounds=cfg.save_every_n_rounds,
        evaluate_client_every_n_rounds=cfg.evaluate_client_every_n_rounds,
        evaluate_server_every_n_rounds=cfg.evaluate_server_every_n_rounds,
        dryrun=cfg.dryrun,
    )

    # Specify Ray config
    if "cpus_per_task" in HydraConfig.get().launcher:  # to launch on slurm cluster
        num_cpus = HydraConfig.get().launcher.cpus_per_task  # Ray bypass the available cpus so we have to specify it
        log(INFO, f"Detecting num_cpus: {num_cpus}")
        ray_init_args = {
            "include_dashboard": False,
            "runtime_env": {"working_dir": get_original_cwd()},
            "num_cpus": num_cpus,
        }
    else:  # local launch
        ray_init_args = {
            "include_dashboard": False,
            "runtime_env": {"working_dir": get_original_cwd()},
        }

    # Client resources
    client_resources = {}

    # Start simulation
    client_manager = SimpleClientManager()
    starting_round = 0
    history = None
    server_init = svr.SavingServer(
        starting_round=starting_round,
        history=history,
        client_manager=client_manager,
        strategy=strategy,
        dryrun=cfg.dryrun,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds_prior),
        strategy=strategy,
        server=server_init,
        ray_init_args=ray_init_args,
    )

    file_suffix: str = (
        f"{cfg.partition_type}"
        f"_C={cfg.num_clients}"
        f"_B={cfg.batch_size_client}"
        f"_E={cfg.num_epochs_prior}"
        f"_R={cfg.num_rounds_prior}"
        f"_P={cfg.prior_type}"
    )

    cwd = os.getcwd()  # Hydra automatically changes the working directory

    np.save(
        Path(cwd) / Path(f"hist{file_suffix}"),
        history,
    )


if __name__ == "__main__":
    OmegaConf.register_new_resolver("my_subdir_suffix", utils.my_subdir_suffix_impl)
    runexp()
