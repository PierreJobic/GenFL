# Federated Learning with Nonvacuous Generalisation Bounds
---

## Introduction

This repository includes the code associated to the paper: "Federated Learning with Nonvacuous Generalisation Bounds"

## Key features

- Train Stochastic Neural Networks with different PAC-Bayes bound objectives through [PyTorch](https://pytorch.org/)
- Simulation of Federated Learning (FL) routine to do collaborative learning thanks to [Flower](https://flower.dev/docs/framework/how-to-install-flower.html)
- Highly configurable experiment thanks to [Hydra](https://hydra.cc/docs/intro/)
- Code is highly parallelisable on CPU/GPU thanks to [PyTorch](https://pytorch.org/), [Flower](https://flower.dev/docs/framework/how-to-install-flower.html), [Ray](https://docs.ray.io/en/master/) and [Slurm](https://slurm.schedmd.com/)

# Quick start guide


This guide will show you how to use `GenFL` for simple applications. Some more detailed informations and more complex scenarios are explained in the `./examples` directory via notebooks.

## Installation

To install and run this project, follow these steps:

1. Clone the repository
2. Install the necessary dependencies


## Code organisation

```bash
(main) me@machine:./PAC_Bayes$ tree -L 1
.
├── conf
├── core
├── examples
├── GenFL_personalized.py
├── GenFL_posterior.py
├── GenFL_prior.py
└── README.md
```
you will find in:
- `conf` every `.yaml` configuration files
- `core` all the important python files
- `examples` some notebooks to show you how to use the code for simple cases
- `GenFL_prior.py`, `GenFL_posterior.py` and `GenFL_personalized.py` to run the code as a CLI

### Special mention: Flower/Hydra

To understand how the federated learning part of the code is made, I advice to check Flower documentation, particularly [this page](https://flower.dev/docs/framework/how-to-implement-strategies.html), the image let us understand how server/client/strategy interact with each others.

Also it is recommended to learn the basics of [Hydra](https://hydra.cc/docs/intro/) to use the code as a CLI.

## Usage

There are 3 main functions:
- `GenFL_prior.py`
- `GenFL_posterior.py`
- `GenFL_personalized.py`

These functions are configurable with `yaml` files which are located in `./conf` directory.  
These 3  functions train neural networks to get (learnt) priors, posteriors and personalized models accordingly.  
Check the `examples` directory to learn how to use it on concrete use cases.

### Examples

In the `examples` directory, you will find notebooks on how to :
- Simple use case :
    - Build a prior 
    - Train a posterior from a random or learnt prior
    - Train personalized posteriors
- More complex use case :
    - How to sweep over parameters
    - How to run on a slurm cluster

# Citation

If you publish work that uses GenFL, please cite the paper : Federated Learning with Nonvacuous Generalisation Bounds

# Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b <branch-name>`
3. Make your changes
4. Push to your fork: `git push origin <branch-name>`
5. Create a pull request

# WIP:
- [x] Train a NN/SNN in a centralized way
- [x] Train a NN/SNN in a FL way
- [x] Random and learnt priors to train posteriors
- [x] Different partitions kind of the dataset to the clients
- [x] Personalized FL
- [x] Running multiple and in parallel experiments on a slurm cluster
- [x] Example notebooks
- [ ] Add a requirements.txt
- [ ] Resuming from a checkpoint
- [ ] Code Documentation