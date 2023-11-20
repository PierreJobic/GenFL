import copy

from typing import Dict, List, Optional, Tuple

import numpy as np

from torch.utils.data import Dataset


def partition_dataset_to_clients(
    trainset: Dataset,
    num_clients: int = 10,
    partition_type: Optional[str] = "exact_iid",
    perc_data: Optional[float] = 1.0,
    seed: Optional[int] = 42,
    logistic: Optional[bool] = False,
) -> Tuple[Dataset, Dataset, List[Dict]]:
    """Split training set into iid or non iid partitions to simulate the
    federated setting.

    Parameters
    ----------
    num_clients : int, optional
        The number of clients that hold a part of the data, by default 10
    iid : bool, optional
        Whether the data should be independent and identically distributed between
        the clients or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario), by default True
    balance : bool, optional
        Whether the dataset should contain an equal number of samples in each class,
        by default True
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Dataset], Dataset]
        A list of dataset for each client and a single dataset to be use for testing the model.

    """

    # Then partition trainset into num_clients partitions
    if partition_type == "exact_iid":
        dict_clients = _partition_exact_iid(dataset=trainset, num_users=num_clients, perc_data=perc_data, seed=seed)
    elif partition_type == "random_iid":
        dict_clients = _partition_random_iid(dataset=trainset, num_users=num_clients, perc_data=perc_data, seed=seed)
    elif partition_type == "random_non_iid":
        dict_clients = _partition_random_non_iid(
            dataset=trainset, num_users=num_clients, perc_data=perc_data, seed=seed
        )
    elif partition_type == "unequal_non_iid":
        dict_clients = _partition_unequal_noniid(
            dataset=trainset, num_users=num_clients, perc_data=perc_data, seed=seed
        )
    else:
        raise ValueError(f"Partition type not recognized: {partition_type}")
    return dict_clients


def partition_class_iid(dataset, num_users, idxs_proxy=None, perc_data=1.0, perc_data_users=None, seed=42):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index

    example
    -------
    100 labels 0
    200 labels 1

    num_users = 2

    50 labels 0 for user 1
    50 labels 0 for user 2
    100 labels 1 for user 1
    100 labels 1 for user 2
    """
    if idxs_proxy is None:
        idxs_proxy = np.arange(len(dataset))
    if perc_data_users is not None:
        assert sum(perc_data_users) == 1.0
        assert num_users == len(perc_data_users)
    else:
        perc_data_users = [1 / num_users] * num_users
    np.random.seed(seed)
    dict_users = {i: set() for i in range(num_users)}
    subset_targets = dataset.targets.numpy()[idxs_proxy]
    class_counts = np.bincount(subset_targets)
    # smallest = np.min(class_counts)
    class_counts = np.insert(class_counts, 0, 0)
    idxs = np.array(subset_targets).argsort()
    class_counts = class_counts[:-1]
    cumsum_class_counts = np.cumsum(class_counts)
    for i in range(len(cumsum_class_counts)):
        count = cumsum_class_counts[i]
        count_next = cumsum_class_counts[i + 1] if i + 1 < len(cumsum_class_counts) else len(idxs)
        available_class_idx = idxs_proxy[idxs[int(count) : int(count_next)]]
        num_items_users = [int((len(available_class_idx) * perc_data_users[i]) * perc_data) for i in range(num_users)]
        for i in range(num_users):
            set_users_i = set(np.random.choice(available_class_idx, num_items_users[i], replace=False))
            dict_users[i] = dict_users[i] | set_users_i
            available_class_idx = list(set(available_class_idx) - set_users_i)
    return dict_users


def _partition_random_iid(dataset, num_users, idxs=None, perc_data=1.0, perc_data_users=None, seed=42):
    np.random.seed(seed)
    if idxs is None:
        idxs = np.arange(len(dataset))
    if perc_data_users is not None:
        assert sum(perc_data_users) == 1.0
        assert num_users == len(perc_data_users)
        num_items_users = [int((len(idxs) * perc_data_user) * perc_data) for perc_data_user in perc_data_users]
    else:
        num_items_users = [int((len(idxs) / num_users) * perc_data)] * num_users

    dict_users = {}
    all_idxs = copy.deepcopy(idxs)
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items_users[i], replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def _partition_exact_iid(dataset, num_users, idxs_proxy=None, perc_data=1.0, perc_data_users=None, seed=42):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index

    example
    -------
    100 labels 0
    200 labels 1

    num_users = 2

    50 labels 0 for user 1
    50 labels 0 for user 2
    50 labels 1 for user 1
    50 labels 1 for user 2

    """
    if idxs_proxy is None:
        idxs_proxy = np.arange(len(dataset))
    if perc_data_users is not None:
        assert sum(perc_data_users) == 1.0
        assert num_users == len(perc_data_users)
    else:
        perc_data_users = [1 / num_users] * num_users
    np.random.seed(seed)
    dict_users = {i: set() for i in range(num_users)}
    subset_targets = dataset.targets.numpy()[idxs_proxy]
    class_counts = np.bincount(subset_targets)
    smallest = np.min(class_counts)
    class_counts = np.insert(class_counts, 0, 0)
    idxs = np.array(subset_targets).argsort()
    class_counts = class_counts[:-1]
    for count in np.cumsum(class_counts):
        available_class_idx = idxs_proxy[idxs[int(count) : int(count + smallest)]]
        num_items_users = [int((len(available_class_idx) * perc_data_users[i]) * perc_data) for i in range(num_users)]
        for i in range(num_users):
            set_users_i = set(np.random.choice(available_class_idx, num_items_users[i], replace=False))
            dict_users[i] = dict_users[i] | set_users_i
            available_class_idx = list(set(available_class_idx) - set_users_i)
    return dict_users


def _partition_random_non_iid(dataset, num_users, perc_data=1.0, seed=42):
    """
    Code taken from: https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/sampling.py
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    np.random.seed(seed)
    # 60,000 training imgs -->  300*perc_data imgs/shard X 200 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: set() for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = dict_users[i] | set(idxs[rand * num_imgs : (rand * num_imgs) + int(num_imgs * perc_data)])
    return dict_users


def _partition_unequal_noniid(dataset, num_users, perc_data=1.0, seed=42):
    """
    Code taken from: https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/sampling.py
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    np.random.seed(seed)
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1, size=num_users)
    random_shard_size = np.around(random_shard_size / sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:
        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : rand * num_imgs + int(num_imgs * perc_data)]), axis=0
                )

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : rand * num_imgs + int(num_imgs * perc_data)]), axis=0
                )
    else:
        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : rand * num_imgs + int(num_imgs * perc_data)]), axis=0
                )

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs : rand * num_imgs + int(num_imgs * perc_data)]), axis=0
                )

    return dict_users
