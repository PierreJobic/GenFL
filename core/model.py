import copy
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")


class Gaussian(nn.Module):
    # Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py
    """Implementation of a Gaussian random variable, using softplus for
    the standard deviation and with implementation of sampling and KL
    divergence computation.

    Parameters
    ----------
    mu : Tensor of floats
        Centers of the Gaussian.

    rho : Tensor of floats
        Scale parameter of the Gaussian (to be transformed to std
        via the softplus function)

    device : string
        Device the code will run in (e.g. 'cuda')

    fixed_mu : bool
        Boolean indicating whether the Gaussian is supposed to be fixed_mu
        or learnt.

    """

    def __init__(self, mu, rho, device=DEVICE, fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)
        self.device = device

    @property
    def sigma(self):
        # Computation of standard deviation:
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. Specifically, we use sigma = log(exp(rho)+1)
        return torch.log(1 + torch.exp(self.rho))

    def sample(self):
        # Return a sample from the Gaussian distribution
        return self.mu + self.sigma * torch.randn_like(self.sigma)

    def compute_kl(self, other):
        # Compute KL divergence between two Gaussians (self and other)
        # (refer to the paper)
        # b is the variance of priors
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other.sigma, 2)

        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(torch.pow(self.mu - other.mu, 2), b0)
        term3 = torch.div(b1, b0)
        kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()
        return kl_div


class GaussianDziugaite(nn.Module):
    def __init__(self, mu, rho, device=DEVICE, fixed_mu=False, fixed_rho=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed_mu)
        self.rho = nn.Parameter(rho, requires_grad=not fixed_rho)  # rho can be either scaler or vector
        self.device = device

    @property
    def sigma(self):
        # Computation of standard deviation:
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. Specifically, we use sigma = log(exp(rho)+1)
        return torch.log(1 + torch.exp(self.rho))
        # return torch.exp(self.rho)

    def sample(self):
        # Return a sample from the Gaussian distribution
        return self.mu + self.sigma * torch.randn_like(self.sigma)

    def compute_kl(self, other_mu, other_sigma):
        # Compute KL divergence between two Gaussians (self and other)
        # (refer to the paper)
        # b is the variance of priors
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other_sigma, 2)

        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(torch.pow(self.mu - other_mu, 2), b0)
        term3 = torch.div(b1, b0)
        kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()
        return kl_div


class Laplace(nn.Module):
    # Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py
    """Implementation of a Laplace random variable, using softplus for
    the scale parameter and with implementation of sampling and KL
    divergence computation.

    Parameters
    ----------
    mu : Tensor of floats
        Centers of the Laplace distr.

    rho : Tensor of floats
        Scale parameter for the distribution (to be transformed
        via the softplus function)

    device : string
        Device the code will run in (e.g. 'cuda')

    fixed_mu : bool
        Boolean indicating whether the distribution is supposed to be fixed_mu
        or learnt.

    """

    def __init__(self, mu, rho, device=DEVICE, fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)
        self.device = device

    @property
    def scale(self):
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. We use sigma = log(exp(rho)+1)
        m = nn.Softplus()
        return m(self.rho)

    def sample(self):
        # Return a sample from the Laplace distribution
        # we do scaling due to numerical issues
        epsilon = (0.999 * torch.rand(self.scale.size()) - 0.49999).to(self.device)
        result = self.mu - torch.mul(torch.mul(self.scale, torch.sign(epsilon)), torch.log(1 - 2 * torch.abs(epsilon)))
        return result

    def compute_kl(self, other):
        # Compute KL divergence between two Laplaces distr. (self and other)
        # (refer to the paper)
        # b is the variance of priors
        b1 = self.scale
        b0 = other.scale
        term1 = torch.log(torch.div(b0, b1))
        aux = torch.abs(self.mu - other.mu)
        term2 = torch.div(aux, b0)
        term3 = torch.div(b1, b0) * torch.exp(torch.div(-aux, b1))

        kl_div = (term1 + term2 + term3 - 1).sum()
        return kl_div


class LinearPerez(nn.Module):
    # Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py
    """Implementation of a Linear layer (reimplemented to use
    truncated normal as initialisation for fair comparison purposes)

    Parameters
    ----------
    in_features : int
        Number of input features for the layer

    out_features : int
        Number of output features for the layer

    device : string
        Device the code will run in (e.g. 'cuda')

    """

    def __init__(self, in_features, out_features, device=DEVICE):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1 / np.sqrt(in_features)

        # same initialisation as before for the prob layer
        self.weight = nn.Parameter(
            trunc_normal_(
                torch.Tensor(out_features, in_features), 0, sigma_weights, -2 * sigma_weights, 2 * sigma_weights
            ),
            requires_grad=True,
        )
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        return F.linear(input, weight, bias)


class LinearDziugaite(nn.Module):
    # Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py
    """Implementation of a Linear layer (reimplemented to use
    truncated normal as initialisation for fair comparison purposes)

    Parameters
    ----------
    in_features : int
        Number of input features for the layer

    out_features : int
        Number of output features for the layer

    device : string
        Device the code will run in (e.g. 'cuda')

    """

    def __init__(self, in_features, out_features, is_first=False, device=DEVICE):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 0.04

        # same initialisation as before for the prob layer
        self.weight = nn.Parameter(
            trunc_normal_(
                torch.Tensor(out_features, in_features), 0, sigma_weights, -2 * sigma_weights, 2 * sigma_weights
            ),
            requires_grad=True,
        )
        if is_first:
            self.bias = nn.Parameter(torch.zeros(out_features) + 0.1, requires_grad=True)
        else:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        return F.linear(input, weight, bias)


class ProbLinearPerez(nn.Module):
    """
    Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py

    Implementation of a Probabilistic Linear layer.

    Parameters
    ----------
    in_features : int
        Number of input features for the layer

    out_features : int
        Number of output features for the layer

    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_layer : Linear object
        Linear layer object used to initialise the prior

    init_prior : string
        string that indicates the way to initialise the prior:
        *"weights" = initialise with init_layer
        *"zeros" = initialise with zeros and rho prior
        *"random" = initialise with random weights and rho prior
        *""

    """

    def __init__(
        self,
        in_features,
        out_features,
        rho_prior,
        prior_dist="gaussian",
        device=DEVICE,
        init_prior="weights",
        init_layer=None,
        init_layer_prior=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1 / np.sqrt(in_features)

        # Posterior initialisation
        if init_layer:
            weights_mu_init = init_layer.weight
            bias_mu_init = init_layer.bias
        else:
            # Initialise distribution means using truncated normal
            weights_mu_init = trunc_normal_(
                torch.Tensor(out_features, in_features), 0, sigma_weights, -2 * sigma_weights, 2 * sigma_weights
            )
            bias_mu_init = torch.zeros(out_features)

        weights_rho_init = torch.ones(out_features, in_features) * rho_prior
        bias_rho_init = torch.ones(out_features) * rho_prior

        # Prior initialisation
        if init_prior == "zeros":
            bias_mu_prior = torch.zeros(out_features)
            weights_mu_prior = torch.zeros(out_features, in_features)
        elif init_prior == "random":
            weights_mu_prior = trunc_normal_(
                torch.Tensor(out_features, in_features), 0, sigma_weights, -2 * sigma_weights, 2 * sigma_weights
            )
            bias_mu_prior = torch.zeros(out_features)
        elif init_prior == "weights":
            if init_layer_prior:
                weights_mu_prior = init_layer_prior.weight
                bias_mu_prior = init_layer_prior.bias
            else:
                # otherwise initialise to posterior weights
                weights_mu_prior = weights_mu_init
                bias_mu_prior = bias_mu_init
        else:
            raise RuntimeError("Wrong type of prior initialisation!")

        # Distribution type
        if prior_dist == "gaussian":
            dist = Gaussian
        elif prior_dist == "laplace":
            dist = Laplace
        else:
            raise RuntimeError(f"Wrong prior_dist {prior_dist}")

        self.weight = dist(copy.deepcopy(weights_mu_init), copy.deepcopy(weights_rho_init), device=device, fixed=False)
        self.bias = dist(copy.deepcopy(bias_mu_init), copy.deepcopy(bias_rho_init), device=device, fixed=False)
        self.weight_prior = dist(
            copy.deepcopy(weights_mu_prior),
            copy.deepcopy(weights_rho_init),
            device=device,
            fixed=True,
        )
        self.bias_prior = dist(
            copy.deepcopy(bias_mu_prior),
            copy.deepcopy(bias_rho_init),
            device=device,
            fixed=True,
        )

    def kl_div(self):
        # KL as a sum of the KL computed for weights and biases
        return self.weight.compute_kl(self.weight_prior) + self.bias.compute_kl(self.bias_prior)

    def forward(self, input, sample=False):
        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            # otherwise we use the posterior mean
            weight = self.weight.mu
            bias = self.bias.mu
        return F.linear(input, weight, bias)


class ProbLinearDziugaite(nn.Module):
    """
    Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py

    Implementation of a Probabilistic Linear layer.

    Parameters
    ----------
    in_features : int
        Number of input features for the layer

    out_features : int
        Number of output features for the layer

    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_layer : Linear object
        Linear layer object used to initialise the prior

    init_prior : string
        string that indicates the way to initialise the prior:
        *"weights" = initialise with init_layer
        *"zeros" = initialise with zeros and rho prior
        *"random" = initialise with random weights and rho prior
        *""

    """

    def __init__(
        self,
        in_features,
        out_features,
        init_layer,
        device=DEVICE,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        # sigma_weights = 0.04

        # Posterior initialisation
        # if init_layer:
        weights_mu_init = init_layer.weight
        bias_mu_init = init_layer.bias
        # else:
        #     # Initialise distribution means using truncated normal
        #     weights_mu_init = trunc_normal_(
        #         torch.Tensor(out_features, in_features), 0, sigma_weights, -2 * sigma_weights, 2 * sigma_weights
        #     )
        #     bias_mu_init = torch.zeros(out_features)

        init_layer_weight = torch.pow(torch.abs(init_layer.weight), 1.0 / 2.0)  # transform variance to std
        init_layer_bias = torch.pow(torch.abs(init_layer.bias), 1.0 / 2.0)

        weights_lambda_init = torch.log(torch.exp(init_layer_weight) - 1)  # transform std to rho
        bias_lambda_init = torch.log(torch.exp(init_layer_bias) - 1)

        # weights_lambda_init = 1.0 / 2.0 * torch.log(torch.abs(init_layer.weight))
        # bias_lambda_init = 1.0 / 2.0 * torch.log(torch.abs(init_layer.bias))

        # Distribution type
        dist = GaussianDziugaite

        self.weight = dist(
            copy.deepcopy(weights_mu_init.detach()), copy.deepcopy(weights_lambda_init.detach()), device=device
        )
        self.bias = dist(copy.deepcopy(bias_mu_init.detach()), copy.deepcopy(bias_lambda_init.detach()), device=device)

    def kl_div(self, other_weight_mu, other_weight_sigma, other_bias_mu, other_bias_sigma):
        # KL as a sum of the KL computed for weights and biases
        weight_kl = self.weight.compute_kl(other_weight_mu, other_weight_sigma)
        bias_kl = self.bias.compute_kl(other_bias_mu, other_bias_sigma)
        return weight_kl + bias_kl

    def forward(self, input, sample=False):
        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            # otherwise we use the posterior mean
            weight = self.weight.mu
            bias = self.bias.mu
        return F.linear(input, weight, bias)


class NNet4lPerez(nn.Module):
    """
    Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py

    Implementation of a standard Neural Network with 4 layers and dropout
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)

    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.

    device : string
        Device the code will run in (e.g. 'cuda')

    """

    def __init__(self, dropout_prob=0.0, device=DEVICE):
        super().__init__()
        self.l1 = LinearPerez(28 * 28, 600, device=device)
        self.l2 = LinearPerez(600, 600, device=device)
        self.l3 = LinearPerez(600, 600, device=device)
        self.l4 = LinearPerez(600, 10, device=device)
        self.d = nn.Dropout(dropout_prob)

    def forward(self, x):
        # forward pass for the network
        x = x.view(-1, 28 * 28)
        x = self.d(self.l1(x))
        x = F.relu(x)
        x = self.d(self.l2(x))
        x = F.relu(x)
        x = self.d(self.l3(x))
        x = F.relu(x)
        x = self.l4(x)
        x = _output_transform(x, clamping=False)
        return x


class NNet4lDziugaite(nn.Module):
    """
    Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py

    Implementation of a standard Neural Network with 4 layers and dropout
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)

    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.

    device : string
        Device the code will run in (e.g. 'cuda')

    """

    def __init__(self, dropout_prob=0.0, device=DEVICE):
        super().__init__()
        self.l1 = LinearDziugaite(28 * 28, 600, is_first=True, device=device)
        self.l2 = LinearDziugaite(600, 600, device=device)
        self.l3 = LinearDziugaite(600, 600, device=device)
        self.l4 = LinearDziugaite(600, 1, device=device)  # It is a binary logistic regression task
        self.d = nn.Dropout(dropout_prob)
        for i, layer in enumerate([self.l1, self.l2, self.l3, self.l4]):  # this is w_0 in the paper
            self.register_buffer(f"l{i+1}_prior_weight", copy.deepcopy(layer.weight.detach()))
            self.register_buffer(f"l{i+1}_prior_bias", copy.deepcopy(layer.bias.detach()))

    def forward(self, x):
        # forward pass for the network
        x = x.view(-1, 28 * 28)
        x = self.d(self.l1(x))
        x = F.relu(x)
        x = self.d(self.l2(x))
        x = F.relu(x)
        x = self.d(self.l3(x))
        x = F.relu(x)
        x = self.l4(x)
        # x = _output_transform(x, clamping=False)
        return x


class ProbNNet4lPerez(nn.Module):
    """
    Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py

    Implementation of a Probabilistic Neural Network with 4 layers
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)

    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_net : NNet object
        Network object used to initialise the prior

    """

    def __init__(self, rho_prior, prior_dist="gaussian", device=DEVICE, init_net=None):
        super().__init__()
        self.l1 = ProbLinearPerez(
            28 * 28, 600, rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.l1 if init_net else None
        )
        self.l2 = ProbLinearPerez(
            600, 600, rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.l2 if init_net else None
        )
        self.l3 = ProbLinearPerez(
            600, 600, rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.l3 if init_net else None
        )
        self.l4 = ProbLinearPerez(
            600, 10, rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.l4 if init_net else None
        )

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.relu(self.l3(x, sample))
        x = self.l4(x, sample)
        x = _output_transform(x, clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.l1.kl_div() + self.l2.kl_div() + self.l3.kl_div() + self.l4.kl_div()


class ProbNNet4lDziugaite(nn.Module):
    """
    Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py

    Implementation of a Probabilistic Neural Network with 4 layers
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)

    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_net : NNet object
        Network object used to initialise the prior

    """

    def __init__(
        self,
        rho_prior,  # Lambda in the paper
        init_net,  # Should be mandatory
        device=DEVICE,
    ):
        super().__init__()
        assert isinstance(init_net, NNet4lDziugaite)
        self.l1 = ProbLinearDziugaite(
            28 * 28,
            600,
            device=device,
            init_layer=init_net.l1,
        )
        self.l2 = ProbLinearDziugaite(
            600,
            600,
            device=device,
            init_layer=init_net.l2,
        )
        self.l3 = ProbLinearDziugaite(
            600,
            600,
            device=device,
            init_layer=init_net.l3,
        )
        self.l4 = ProbLinearDziugaite(
            600,
            1,  # It is a binary task
            device=device,
            init_layer=init_net.l4,
        )

        # This next line is very important to save and load w_0 through the state_dict
        # For instance it is useful when server send parameters to clients
        # Or to restore a model from a checkpoint without losing prior information
        for i in range(1, 5):  # this is w_0 in the paper
            self.register_buffer(f"l{i}_prior_weight", getattr(init_net, f"l{i}_prior_weight"))
            self.register_buffer(f"l{i}_prior_bias", getattr(init_net, f"l{i}_prior_bias"))

        self.rho_prior = nn.Parameter(torch.tensor(rho_prior), requires_grad=True)

    @property
    def sigma_prior(self):
        # Computation of standard deviation:
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. Specifically, we use sigma = exp(rho)
        return torch.exp(self.rho_prior)

    def forward(self, x, sample=False, clamping=False, pmin=1e-4):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.relu(self.l3(x, sample))
        x = self.l4(x, sample)
        # x = _output_transform(x, False, None)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        # rho_prior needs to be a leaf tensor that is optimised during training
        # that is why we can't just seperate it into different Gaussian Modules
        # Hence we have to manually compute it
        l1_kl_div = self.l1.kl_div(
            other_weight_mu=self.l1_prior_weight,
            other_weight_sigma=self.sigma_prior,
            other_bias_mu=self.l1_prior_bias,
            other_bias_sigma=self.sigma_prior,
        )
        l2_kl_div = self.l2.kl_div(
            other_weight_mu=self.l2_prior_weight,
            other_weight_sigma=self.sigma_prior,
            other_bias_mu=self.l2_prior_bias,
            other_bias_sigma=self.sigma_prior,
        )
        l3_kl_div = self.l3.kl_div(
            other_weight_mu=self.l3_prior_weight,
            other_weight_sigma=self.sigma_prior,
            other_bias_mu=self.l3_prior_bias,
            other_bias_sigma=self.sigma_prior,
        )
        l4_kl_div = self.l4.kl_div(
            other_weight_mu=self.l4_prior_weight,
            other_weight_sigma=self.sigma_prior,
            other_bias_mu=self.l4_prior_bias,
            other_bias_sigma=self.sigma_prior,
        )
        return l1_kl_div + l2_kl_div + l3_kl_div + l4_kl_div


# ## Utils NN ## #
def _output_transform(x, clamping=True, pmin=1e-4):
    # Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py#L1097
    """Computes the log softmax and clamps the values using the
    min probability given by pmin.

    Parameters
    ----------
    x : tensor
        output of the network

    clamping : bool
        whether to clamp the output probabilities

    pmin : float
        threshold of probabilities to clamp.
    """
    # lower bound output prob
    output = F.log_softmax(x, dim=1)
    if clamping:
        output = torch.clamp(output, np.log(pmin))
    return output


def trunc_normal_network_(network, mean=0.0, std=1.0, a=-2.0, b=2.0, only_first=False):
    # Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py
    is_first = True
    with torch.no_grad():
        for name, params in network.named_parameters():
            if "weight" in name:
                params.data = trunc_normal_(params.data, mean, std, a, b)
            elif "bias" in name:
                if only_first:
                    if is_first:
                        is_first = False
                        params.data = torch.zeros(params.size()) + 0.1
                    else:
                        params.data = torch.zeros(params.size())
                else:
                    params.data = torch.zeros(params.size()) + 0.1


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used works best if :math:`\text{mean}` is
    near the center of the interval.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Code taken from: https://github.com/mperezortiz/PBB/blob/master/pbb/models.py
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Fill tensor with uniform values from [l, u]
        tensor.uniform_(lower, upper)

        # Use inverse cdf transform from normal distribution
        tensor.mul_(2)
        tensor.sub_(1)

        # Ensure that the values are strictly between -1 and 1 for erfinv
        eps = torch.finfo(tensor.dtype).eps
        tensor.clamp_(min=-(1.0 - eps), max=(1.0 - eps))
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp one last time to ensure it's still in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
