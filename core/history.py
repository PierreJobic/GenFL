import os
import copy
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from functools import reduce
from typing import Optional
from logging import WARNING

from flwr.server.history import History
from flwr.common.logger import log
from . import utils

ATTR_NAMES = [
    "losses_centralized",
    "losses_distributed",
    "metrics_centralized",
    "metrics_distributed",
    "metrics_distributed_fit",
]


class CustomHistory(History):
    def __repr__(self) -> str:
        rep = ""
        if self.losses_distributed:
            rep += "History (loss, distributed):\n" + reduce(
                lambda a, b: a + b,
                [f"\tround {server_round}: {loss}\n" for server_round, loss in self.losses_distributed],
            )
        if self.losses_centralized:
            rep += "History (loss, centralized):\n" + reduce(
                lambda a, b: a + b,
                [f"\tround {server_round}: {loss}\n" for server_round, loss in self.losses_centralized],
            )
        return rep


# ## History Management Functions ## #
def plot_all_metrics_from_history(
    hist: History,
    save_plot_path: Path,
    suffix: Optional[str] = "",
) -> None:
    metrics_centralized_name = hist.metrics_centralized.keys()
    metrics_distributed_name = hist.metrics_distributed.keys()
    metrics_distributed_fit_name = hist.metrics_distributed_fit.keys()
    all_metrics_name = list(
        set(metrics_centralized_name) | set(metrics_distributed_name) | set(metrics_distributed_fit_name)
    )

    save_plot_path = save_plot_path + "/metrics"
    if not os.path.exists(save_plot_path):
        os.mkdir(save_plot_path)

    # Losses plot
    _ = plt.figure(figsize=(20, 10))
    if hist.losses_distributed != []:
        rounds_distributed, values_distributed = zip(*hist.losses_distributed)
        plt.plot(np.asarray(rounds_distributed), np.asarray(values_distributed), label="distributed")
    if hist.losses_centralized != []:
        rounds_centralized, values_centralized = zip(*hist.losses_centralized)
        plt.plot(np.asarray(rounds_centralized), np.asarray(values_centralized), label="centralized")
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title(f"Losses (Distributed and Centralized) with {suffix}")
    plt.legend()
    plt.savefig(Path(save_plot_path) / Path(f"Losses_metrics{suffix}.png"))
    plt.close()

    # Metrics plot
    for metric_name in all_metrics_name:
        _ = plt.figure(figsize=(20, 10))
        if metric_name in metrics_centralized_name:
            rounds, values = zip(*hist.metrics_centralized[metric_name])
            plt.plot(
                np.asarray(rounds),
                np.asarray(values),
                label="centralized",
            )
        if metric_name in metrics_distributed_name:
            rounds, values = zip(*hist.metrics_distributed[metric_name])
            plt.plot(
                np.asarray(rounds),
                np.asarray(values),
                label="distributed",
            )
        if metric_name in metrics_distributed_fit_name:
            rounds, values = zip(*hist.metrics_distributed_fit[metric_name])
            plt.plot(
                np.asarray(rounds),
                np.asarray(values),
                label="distributed_fit",
            )
        plt.xlabel("Rounds")
        plt.ylabel("Metrics")
        plt.title(f"Metrics {metric_name}: (Distributed, Distributed Fit and Centralized) with {suffix}")
        plt.legend()
        plt.savefig(Path(save_plot_path) / Path(f"{metric_name}_Metrics{suffix}.png"))
        plt.close()

    # Save History
    np.save(save_plot_path + "/hist_tmp", hist)


def get_history(path_to_checkpoint_dir, round):
    history_path = utils.get_last_from_pattern(path_to_checkpoint_dir + "/metrics/hist_*.npy")
    history = np.load(history_path, allow_pickle=True).item()

    # Check round information
    max_round_history = max(
        history.losses_distributed[-1][0],
        history.losses_centralized[-1][0],
    )

    if round != max_round_history:
        log(
            WARNING,
            f"max round in history {max_round_history} is different from loaded round: {round}",
        )
        log(WARNING, f"Cutting history to match the loaded round: {round}")
        history = _cutted_history(history, on_round=round)
    return history


def _reversed_cutted_history(hist, on_round, make_copy=True):
    """Cut the history at a specific round to keep only the metrics before that round."""
    _hist = copy.deepcopy(hist) if make_copy else hist
    for attr_name in ATTR_NAMES:
        attr = getattr(_hist, attr_name)
        if attr is not None:
            if isinstance(attr) == list:
                if len(attr) > on_round:
                    setattr(_hist, attr_name, attr[on_round:])
            elif isinstance(attr) == dict:
                for key in attr.keys():
                    if len(attr[key]) > on_round:
                        attr[key] = attr[key][on_round:]
    return _hist


def _cutted_history(hist, on_round, make_copy=True):
    """Cut the history at a specific round to keep only the metrics before that round."""
    _hist = copy.deepcopy(hist) if make_copy else hist
    for attr_name in ATTR_NAMES:
        attr = getattr(_hist, attr_name)
        if attr is not None:
            if isinstance(attr) == list:
                setattr(_hist, attr_name, [(round, _loss) for round, _loss in attr if round <= on_round])
            elif isinstance(attr) == dict:
                for key in attr.keys():
                    attr[key] = [(round, metric) for round, metric in attr[key] if round <= on_round]
    return _hist


def _shifted_history(hist, shift_to_round, make_copy=True):
    """Shift the history by a specific number of rounds."""
    _hist = copy.deepcopy(hist) if make_copy else hist
    for attr_name in ATTR_NAMES:
        attr = getattr(_hist, attr_name)
        if attr is not None:
            if isinstance(attr) == list:
                setattr(_hist, attr_name, [(rounds + shift_to_round + 1, attr_value) for rounds, attr_value in attr])
            elif isinstance(attr) == dict:
                for key in attr.keys():
                    attr[key] = [(rounds + shift_to_round + 1, attr_value) for rounds, attr_value in attr[key]]
    return _hist


def _concatenated_history(hist_1, hist_2, on_round):
    """Concatenate two histories on a specific round."""
    hist_1_cutted = _cutted_history(hist_1, on_round, make_copy=True)
    hist_2_shifted = _shifted_history(hist_2, on_round, make_copy=True)
    for attr_name in ATTR_NAMES:
        attr_1 = getattr(hist_1_cutted, attr_name)
        attr_2 = getattr(hist_2_shifted, attr_name)
        if attr_1 is not None and attr_2 is not None:
            if isinstance(attr_1) == list:
                setattr(hist_1_cutted, attr_name, attr_1 + attr_2)
            elif isinstance(attr_1) == dict:
                for key in attr_1.keys():
                    attr_1[key] = attr_1[key] + attr_2[key]
    return hist_1_cutted
