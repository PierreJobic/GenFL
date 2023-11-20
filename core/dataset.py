"""
Some code are taken from https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/sampling.py
"""
# from logging import INFO, DEBUG
# from flwr.common.logger import log

from typing import Optional, Tuple

import numpy as np

# import torch
import torchvision


from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

from . import partition

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

MNIST_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)])
TARGET_MNIST_LOGISTIC_TRANSFORM = transforms.Compose(
    [
        transforms.Lambda(lambda x: 1 if x >= 5 else -1),
    ]
)


def load_datasets(
    data_path: str = "~/data",
    num_clients: int = 10,
    partition_type: Optional[str] = "exact_iid",
    batch_size_client: Optional[int] = 32,
    batch_size_server: Optional[int] = 32,
    seed: Optional[int] = 42,
    logistic: Optional[bool] = False,
    perc_data: Optional[float] = 1.0,
    perc_val: float = 0.0,
    perc_test: Optional[float] = 0.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates the dataloaders to be fed into the model.

    Parameters
    ----------
    num_clients : int, optional
        The number of clients that hold a part of the data, by default 10
    iid : bool, optional
        Whether the data should be independent and identically distributed between the
        clients or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario), by default True
    balance : bool, optional
        Whether the dataset should contain an equal number of samples in each class,
        by default True
    perc_val : float, optional
        The ratio of training data that will be used for validation (between 0 and 1),
        by default 0.1
    batch_size : int, optional
        The size of the batches to be fed into the model, by default 32
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoader for training, the DataLoader for validation, the DataLoader for testing.
    """
    trainset, testset = _get_trainset_and_testset(data_path, logistic=logistic)
    dict_clients = partition.partition_dataset_to_clients(
        trainset, num_clients, partition_type, perc_data, seed, logistic=logistic
    )
    trainloaders = []
    whole_trains = []
    valloaders = []
    testloaders = []
    assert perc_val + perc_test <= 1.0
    train_ratio = 1.0 - perc_val - perc_test
    for _, dict_client in dict_clients.items():
        # Split the client's dataset into train, val and test
        dict_dataset = partition.partition_class_iid(
            trainset,
            3,
            idxs_proxy=np.array(list(dict_client)),
            perc_data_users=np.array([train_ratio, perc_val, perc_test]),
            seed=seed,
        )
        ds_train, ds_val, ds_test = (
            DatasetSplit(trainset, dict_dataset[0]),
            DatasetSplit(trainset, dict_dataset[1]),
            DatasetSplit(trainset, dict_dataset[2]),
        )
        if len(ds_train) != 0:
            trainloaders.append(
                DataLoader(
                    ds_train,
                    batch_size=batch_size_client,
                    shuffle=True,
                    persistent_workers=True,
                    pin_memory=True,
                    num_workers=2,
                )
            )
            whole_trains.append(DataLoader(ds_train, batch_size=len(ds_train)))
        if len(ds_val) != 0:
            valloaders.append(DataLoader(ds_val, batch_size=batch_size_client))
        if len(ds_test) != 0:
            testloaders.append(DataLoader(ds_test, batch_size=batch_size_client))
    test_loader = DataLoader(testset, batch_size=batch_size_server)
    return (
        trainloaders,
        whole_trains,
        valloaders,
        testloaders,
        test_loader,
    )


def _get_trainset_and_testset(data_path, logistic: Optional[bool] = False) -> Tuple[Dataset, Dataset]:
    """Returns the MNIST dataset.

    Returns
    -------
    Tuple[MNIST, MNIST]
        The dataset for training and the dataset for testing MNIST.
    """
    if logistic:
        trainset = _logistic_MNIST(data_path, train=True)
        testset = _logistic_MNIST(data_path, train=False)
    else:
        trainset = MNIST(data_path, train=True, transform=MNIST_TRANSFORM)
        testset = MNIST(data_path, train=False, transform=MNIST_TRANSFORM)
    return trainset, testset


def _logistic_MNIST(data_path, train=True, subindex=55000):
    dataset = torchvision.datasets.MNIST(
        data_path,
        train=train,
        transform=MNIST_TRANSFORM,
        target_transform=TARGET_MNIST_LOGISTIC_TRANSFORM,
    )
    return dataset


class DatasetSplit(Dataset):
    """
    Code taken from: https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/update.py
    An abstract Dataset class wrapped around Pytorch Dataset class in order to facilitate partition through index.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
