from typing import List, Tuple

from flwr.common import Metrics


# ## Utils for Metrics Aggregation Functions ## #
def _weighted_average_metric(metrics, name=None):
    metric = [num_examples * m[name] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    num_examples = float(sum(examples))
    if num_examples == 0:
        return None
    return float(sum(metric)) / num_examples


def _max_metric(metrics, name=None):
    metric = [m[name] for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    num_examples = float(sum(examples))
    if num_examples == 0:
        return None
    return float(max(metric))


def _min_metric(metrics, name=None):
    metric = [m[name] for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    num_examples = float(sum(examples))
    if num_examples == 0:
        return None
    return float(min(metric))


def _sum_metric(metrics, name=None):
    metric = [m[name] for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    num_examples = float(sum(examples))
    if num_examples == 0:
        return None
    return sum(metric)


def _del_none_values(dict):
    return {k: v for k, v in dict.items() if v is not None}


# ## (Fit and Evaluate) Metrics Aggregation Functions ## #
METRICS_FN = {
    "loss": _weighted_average_metric,
    "accuracy": _weighted_average_metric,
    "num_examples": _sum_metric,
    "count": _sum_metric,
    "log_lambda": _weighted_average_metric,
    "A": _weighted_average_metric,
    "B": _weighted_average_metric,
    "loss_mc": _weighted_average_metric,
    "accuracy_mc": _weighted_average_metric,
    "loss_stch": _weighted_average_metric,
    "count_stch": _sum_metric,
    "accuracy_stch": _weighted_average_metric,
    "num_examples_stch": _sum_metric,
    "loss_ens": _weighted_average_metric,
    "count_ens": _sum_metric,
    "accuracy_ens": _weighted_average_metric,
    "num_examples_ens": _sum_metric,
    "error_ce": _weighted_average_metric,
    "error_01": _weighted_average_metric,
    "error_ce_min": _min_metric,
    "error_01_min": _min_metric,
    "error_ce_max": _max_metric,
    "error_01_max": _max_metric,
    "train_obj": _weighted_average_metric,
    "accuracy_mean": _weighted_average_metric,
    "loss_mean": _weighted_average_metric,
    "count_mean": _sum_metric,
    "num_examples_mean": _sum_metric,
    "train_obj_l": _weighted_average_metric,
    "accuracy_l": _weighted_average_metric,
    "KL_term": _weighted_average_metric,
    "KL_n": _weighted_average_metric,
    "NLL_loss": _weighted_average_metric,
    "KL_n_l": _weighted_average_metric,
    "NLL_loss_l": _weighted_average_metric,
    "lambda_var": _weighted_average_metric,
    "train_obj_c": _weighted_average_metric,
    "kl_n_c": _weighted_average_metric,
    "empirical_risk_ce_c": _weighted_average_metric,
    "empirical_risk_01_c": _weighted_average_metric,
    "risk_ce_c": _weighted_average_metric,
    "risk_01_c": _weighted_average_metric,
    "risk_ce_c_min": _min_metric,
    "risk_01_c_min": _min_metric,
    "risk_ce_c_max": _max_metric,
    "risk_01_c_max": _max_metric,
}


def metrics_aggregation_universal_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    metrics_aggregted = {}
    for metric_name in metrics[0][1].keys():
        metrics_aggregted[metric_name] = METRICS_FN[metric_name](metrics, name=metric_name)
    return _del_none_values(metrics_aggregted)


def metrics_aggregation_universal_personalized_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    metrics_aggregted = {}
    for metric_name in metrics[0][1].keys():
        if METRICS_FN[metric_name] == _weighted_average_metric:
            metrics_aggregted[metric_name + "_min"] = _min_metric(metrics, name=metric_name)
            metrics_aggregted[metric_name + "_max"] = _max_metric(metrics, name=metric_name)
        metrics_aggregted[metric_name] = METRICS_FN[metric_name](metrics, name=metric_name)
    return _del_none_values(metrics_aggregted)
