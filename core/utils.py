import glob
import re

from omegaconf import ListConfig

NUMBERS = re.compile(r"(\d+)")


def my_subdir_suffix_impl(
    task_overrides: ListConfig,  # list[str]: overrides passed at command line
    exclude_patterns: ListConfig,  # list[str]: regex patterns to exclude
) -> str:
    """
    Code taken from : https://github.com/facebookresearch/hydra/issues/1873

    Return a sting: concatenation of overrides that are not matched by any of the `exclude_patterns`."""
    import re

    rets: list[str] = []
    for override in task_overrides:
        should_exclude = any(re.search(exc_pat, override) for exc_pat in exclude_patterns)
        if not should_exclude:
            rets.append(override)

    return "_".join(rets)


def numerical_sort(value):
    parts = NUMBERS.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def get_last_from_pattern(path_pattern):
    return sorted(glob.glob(path_pattern), key=numerical_sort)[-1]
