# -------------------------------------------------- #
#       Latent Space Path Minimisation Models        #
#                   Matt Sampson                     #
#                  Peter Melchior                    #
#                                                    #
# Initial LatentODE-RNN architecture modified from   #
# https://arxiv.org/abs/1907.03907, with jax/diffrax #
# implementation initially from Patrick Kidger       #
# -------------------------------------------------- #
import time
import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from numpy._typing import _32Bit
from numpy.lib.shape_base import row_stack
import optax
import os
from jax import config

config.update("jax_enable_x64", True)


# The nn representing the ODE function
class Func(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __call__(self, t, y, args):
        return self.scale * self.mlp(y)


# The LatentODE model based on a Variational Autoencoder
class LatentODE(eqx.Module):
    func: Func
    rnn_cell: eqx.nn.GRUCell

    hidden_to_latent: eqx.nn.Linear
    latent_to_hidden: eqx.nn.MLP
    hidden_to_data: eqx.nn.Linear

    hidden_size: int
    latent_size: int
    alpha: int

    lossType: str

    def __init__(
        self,
        *,
        data_size,
        hidden_size,
        latent_size,
        width_size,
        depth,
        alpha,
        key,
        lossType,
        **kwargs,
    ):
        super().__init__(**kwargs)

        mkey, gkey, hlkey, lhkey, hdkey = jr.split(key, 5)

        scale = jnp.ones(())
        mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=mkey,
        )
        self.func = Func(scale, mlp)
        self.rnn_cell = eqx.nn.GRUCell(data_size + 1, hidden_size, key=gkey)
        # NOTE: the expanding dimension by a factor of 2, emmpirically found to work well for RNN by the authors
        self.hidden_to_latent = eqx.nn.Linear(hidden_size, 2 * latent_size, key=hlkey)
        self.latent_to_hidden = eqx.nn.MLP(
            latent_size, hidden_size, width_size=width_size, depth=depth, key=lhkey
        )
        self.hidden_to_data = eqx.nn.Linear(hidden_size, data_size, key=hdkey)

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.alpha = alpha

        self.lossType = lossType

    # Encoder of the VAE edditted to return context
    def _latent(self, ts, ys, key):
        data = jnp.concatenate([ts[:, None], ys], axis=1)
        hidden = jnp.zeros((self.hidden_size,))
        for data_i in reversed(data):
            hidden = self.rnn_cell(data_i, hidden)
        context = self.hidden_to_latent(hidden)
        mean, logstd = context[: self.latent_size], context[self.latent_size :]
        std = jnp.exp(logstd)
        latent = mean + jr.normal(key, (self.latent_size,)) * std
        return latent, mean, std, context

    # Decoder of the VAE
    def _sample(self, ts, latent):
        dt0 = 0.125  # selected as a reasonable choice for this problem
        y0 = self.latent_to_hidden(latent)
        solver = (
            diffrax.Tsit5()
        )  # see: https://docs.kidger.site/diffrax/api/solvers/ode_solvers/
        adjoint = (
            diffrax.RecursiveCheckpointAdjoint()
        )  # see: https://docs.kidger.site/diffrax/api/adjoints/
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            solver,
            ts[0],
            ts[-1],
            dt0,
            y0,
            saveat=diffrax.SaveAt(ts=ts),
            adjoint=adjoint,
        )
        return jax.vmap(self.hidden_to_data)(sol.ys)

    # Standard LatentODE-RNN loss as in https://arxiv.org/abs/1907.03907
    @staticmethod
    def _loss(ys, pred_ys, mean, std):
        # -log p_θ with Gaussian p_θ
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
        # KL(N(mean, std^2) || N(0, 1))
        variational_loss = 0.5 * jnp.sum(mean**2 + std**2 - 2 * jnp.log(std) - 1)
        return reconstruction_loss + variational_loss

    # Standard loss plus path penanlty
    @staticmethod
    def _pathpenaltyloss(self, ys, pred_ys, pred_latent, mean, std):
        # -log p_θ with Gaussian p_θ
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
        # KL(N(mean, std^2) || N(0, 1))
        variational_loss = 0.5 * jnp.sum(mean**2 + std**2 - 2 * jnp.log(std) - 1)
        # Mahalanobis distance between latents \sqrt{(x - y)^T \Sigma^{-1} (x - y)}
        diff = jnp.diff(pred_latent, axis=0)
        std_latent = self.hidden_to_latent(
            self.latent_to_hidden(std)
        )  # get the latent space std
        Cov = jnp.eye(diff.shape[1]) * std_latent  # latent_state
        Cov = jnp.linalg.inv(Cov)
        d_latent = jnp.sqrt(jnp.abs(jnp.sum(jnp.dot(diff, Cov) @ diff.T, axis=1)))
        d_latent = jnp.sum(d_latent)
        alpha = self.alpha  # weighting parameter for distance penalty
        return reconstruction_loss + variational_loss + alpha * d_latent

    # training routine with suite of 3 loss functions
    def train(self, ts, ys, *, key):
        latent, mean, std, context = self._latent(ts, ys, key)
        pred_ys = self._sample(ts, latent)
        pred_latent = self._sampleLatent(ts, latent)
        # the classic VAE based LatentODE-RNN from https://arxiv.org/abs/1907.03907
        if self.lossType == "default":
            return self._loss(ys, pred_ys, mean, std)
        # the classic LatentODE-RNN with the path length penalty
        elif self.lossType == "mahalanobis":
            return self._pathpenaltyloss(self, ys, pred_ys, pred_latent, mean, std)
        else:
            raise ValueError("lossType must be one of 'default' or 'mahalanobis'")

    # Run just the decoder during inference.
    def sample(self, ts, *, key):
        latent = jr.normal(key, (self.latent_size,))
        return self._sample(ts, latent)

    def _sampleLatent(self, ts, latent):
        dt0 = 0.25  # selected as a reasonable choice for this problem
        y0 = self.latent_to_hidden(latent)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return jax.vmap(self.hidden_to_latent)(sol.ys)

    def sampleLatent(self, ts, *, key):
        latent = jr.normal(key, (self.latent_size,))
        return self._sampleLatent(ts, latent)

    # track the path length
    def pathLength(self, ts, ys, *, key):
        latent, mean, std, context = self._latent(ts, ys, key)
        pred_latent = self._sampleLatent(ts, latent)
        # Mahalanobis distance between latents \sqrt{(x - y)^T \Sigma^{-1} (x - y)}
        diff = jnp.diff(pred_latent, axis=0)
        std_latent = self.hidden_to_latent(
            self.latent_to_hidden(std)
        )  # get the latent space std
        Cov = jnp.eye(diff.shape[1]) * std_latent  # latent_state
        Cov = jnp.linalg.inv(Cov)
        d_latent = jnp.sqrt(jnp.abs(jnp.sum(jnp.dot(diff, Cov) @ diff.T, axis=1)))
        d_latent = jnp.sum(d_latent)
        return d_latent


def get_data(dataset_size, *, key, bounds, t_end=1, n_points=100, IC_min=1, IC_max=2):
    ykey, tkey1, tkey2 = jr.split(key, 3)
    y0 = jr.uniform(
        ykey, (dataset_size, 2), minval=IC_min, maxval=IC_max
    )  # ranomize the ICs
    t0 = 0
    # randomize the total time series between t_end and 2 * t_end (t_end is user defined)
    t1 = t_end + 1 * jr.uniform(tkey1, (dataset_size,), minval=1, maxval=5)
    ts = jr.uniform(tkey2, (dataset_size, n_points)) * (t1[:, None] - t0) + t0
    ts = jnp.sort(ts)
    dt0 = 0.1

    # ------------------------
    # Lotka-Volterra equations
    def LVE(t, y, args):
        prey, predator = y
        a, b, c, d = args
        d_prey = a * prey - b * prey * predator
        d_predator = -d * predator + c * prey * predator
        d_y = jnp.array([d_prey, d_predator])
        return d_y

    def solve(ts, y0, bounds, key):
        a_key, b_key, c_key, d_key = jr.split(key, 4)
        # randomly sample for each value in the bounds 
        alpha = jax.random.uniform(a_key, shape=(1,), minval=bounds[0][0], maxval=bounds[0][1])
        beta = jax.random.uniform(b_key, shape=(1,), minval=bounds[1][0], maxval=bounds[1][1])
        delta = jax.random.uniform(c_key, shape=(1,), minval=bounds[2][0], maxval=bounds[2][1])
        gamma = jax.random.uniform(d_key, shape=(1,), minval=bounds[3][0], maxval=bounds[3][1])
        args = jnp.squeeze(jnp.asarray([alpha, beta, delta, gamma]))
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(LVE),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            args=args,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys, args

    # for now seperate the call to LVE to allow random input params
    key = jax.random.PRNGKey(0)
    key_dataset = jr.split(key, dataset_size)
    # make the bounds the same size as the dataset
    bounds = jnp.repeat(jnp.array(bounds)[None, :], dataset_size, axis=0)   
    ys, params = jax.vmap(solve)(ts, y0, bounds, key_dataset)

    return ts, ys, params, y0


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while start < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size
