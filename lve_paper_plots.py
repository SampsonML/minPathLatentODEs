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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy._typing import _32Bit
from numpy.lib.shape_base import row_stack
import optax
import os
import cmasher as cmr
from matplotlib.gridspec import GridSpec

# turn on float 64 - needed to stabalise diffrax for small gradients (see: https://docs.kidger.site/diffrax/further_details/faq/)
from jax import config

config.update("jax_enable_x64", True)

# Matt's standard plot params - Astro style
# ---------------------------------------------- #
import matplotlib as mpl

mpl.rcParams["xtick.top"] = True
mpl.rcParams["ytick.right"] = True
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["xtick.major.size"] = 7
mpl.rcParams["xtick.minor.size"] = 4.5
mpl.rcParams["ytick.major.size"] = 7
mpl.rcParams["ytick.minor.size"] = 4.5
mpl.rcParams["xtick.major.width"] = 2
mpl.rcParams["xtick.minor.width"] = 1.5
mpl.rcParams["ytick.major.width"] = 2
mpl.rcParams["ytick.minor.width"] = 1.5
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams.update({"text.usetex": True})
# ---------------------------------------------- #


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

    # Encoder of the VAE
    def _latent(self, ts, ys, key):
        data = jnp.concatenate([ts[:, None], ys], axis=1)
        hidden = jnp.zeros((self.hidden_size,))
        for data_i in reversed(data):
            hidden = self.rnn_cell(data_i, hidden)
        context = self.hidden_to_latent(hidden)
        mean, logstd = context[: self.latent_size], context[self.latent_size :]
        std = jnp.exp(logstd)
        latent = mean + jr.normal(key, (self.latent_size,)) * std
        return latent, mean, std

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

    # New loss function, no variational loss
    @staticmethod
    def _distanceloss(self, ys, pred_ys, pred_latent, std):
        # -log p_θ with Gaussian p_θ
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
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
        # penalty for shinking latent space
        magnitude = 1 / jnp.linalg.norm(std_latent)
        distance_loss = alpha * d_latent * magnitude
        return reconstruction_loss + distance_loss

    # training routine with suite of 3 loss functions
    def train(self, ts, ys, *, key):
        latent, mean, std = self._latent(ts, ys, key)
        pred_ys = self._sample(ts, latent)
        pred_latent = self._sampleLatent(ts, latent)
        # the classic VAE based LatentODE-RNN from https://arxiv.org/abs/1907.03907
        if self.lossType == "default":
            return self._loss(ys, pred_ys, mean, std)
        # the classic LatentODE-RNN with the path length penalty
        elif self.lossType == "mahalanobis":
            return self._pathpenaltyloss(self, ys, pred_ys, pred_latent, mean, std)
        # our new autoencoder (not VAE) LatentODE-RNN with no variational loss TODO: test this
        elif self.lossType == "distance":
            return self._distanceloss(self, ys, pred_ys, pred_latent, std)
        else:
            raise ValueError(
                "lossType must be one of 'default', 'mahalanobis', or 'distance'"
            )

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
        latent, mean, std = self._latent(ts, ys, key)
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


def get_data(dataset_size, *, key, func=None, t_end=1, n_points=100):
    ykey, tkey1, tkey2 = jr.split(key, 3)
    # NOTE: the initial conditions are randomised for each dataset by min and max, set manually for now
    IC_min = 1
    IC_max = 4
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

    # for testing purposes, trials use randomised coefs
    LVE_args = (1.5, 1.5, 2.5, 1.5)  # a, b, c, d

    # --------------------------
    # Simple harmonic oscillator
    def SHO(t, y, args):
        y1, y2 = y
        theta = args
        dy1 = y2
        dy2 = -y1 - theta * y2
        d_y = jnp.array([dy1, dy2])
        return d_y

    SHO_args = 0.12  # theta

    # --------------------------------------
    # WaterBucket
    # TODO: implement Peters water model
    def Water(t, y, args):
        pass

    if func == "LVE":
        vector_field = LVE
        bounds = [
            (0.5, 1.5),
            (0.5, 1.5),
            (1.5, 2.5),
            (0.5, 1.5),
        ]  # same as https://arxiv.org/pdf/2105.03835.pdf
        # NOTE: args are randomly samples for each dataset between the bounds
    elif func == "SHO":
        vector_field = SHO
        args = SHO_args  # Fixed damping rate
    else:
        raise ValueError("func must be one of 'LVE', 'SHO', 'water'")

    def solve(ts, y0):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            args=args,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    # Hard coding some things for now to be sure works as expected
    def solveLVE(ts, y0, key):
        bounds = [
            (0.75, 1.25),
            (0.75, 1.25),
            (1.75, 2.25),
            (0.75, 1.25),
        ]  # same as https://arxiv.org/pdf/2105.03835.pdf
        args = tuple(
            jax.random.uniform(key, shape=(1,), minval=lb, maxval=ub)
            for (lb, ub) in bounds
        )
        args = jnp.squeeze(jnp.asarray(args))
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
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
    if func == "LVE":
        key = jax.random.PRNGKey(0)
        key_dataset = jr.split(key, dataset_size)
        ys, args = jax.vmap(solveLVE)(ts, y0, key_dataset)
    else:
        ys = jax.vmap(solve)(ts, y0)

    return ts, ys


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


def main(
    train=True,
    dataset_size=20000,
    batch_size=256,
    n_points=100,
    lr=1e-2,
    steps=30,
    plot_every=10,
    save_every=10,
    error_every=10,
    hidden_size=8,
    latent_size=2,
    width_size=8,
    depth=2,
    alpha=1,
    seed=1992,
    t_final=20,
    lossType="default",
    func="SHO",
    figname="latent_ODE.png",
    save_name="blank",
    MODEL_NAME="blank",
    MODEL_NAME2="blank",
):
    # Defining vector fields again for use in visualisation
    # ------------------------
    # Lotka-Volterra equations
    def LVE(t, y, args):
        prey, predator = y
        a, b, c, d = args
        d_prey = a * prey - b * prey * predator
        d_predator = -d * predator + c * prey * predator
        d_y = jnp.array([d_prey, d_predator])
        return d_y

    LVE_args = (1.6, 1.6, 2.6, 1.6)  # a=prey-growth, b, c, d

    # --------------------------
    # Simple harmonic oscillator
    def SHO(t, y, args):
        y1, y2 = y
        theta = args
        dy1 = y2
        dy2 = -y1 - theta * y2
        d_y = jnp.array([dy1, dy2])
        return d_y

    SHO_args = 0.12  # theta

    if func == "LVE":
        vector_field = LVE
        args = LVE_args
        rows = 4
        TITLE = "Latent ODE Model: Lotka-Volterra Equations"
        LAB_X = "prey"
        LAB_Y = "predator"
    elif func == "SHO":
        vector_field = SHO
        args = SHO_args
        rows = 4
        TITLE = "Latent ODE Model: Simple Harmonic Oscillator"
        LAB_X = "position"
        LAB_Y = "velocity"
    else:
        raise ValueError("func must be one of 'LVE', 'SHO', 'water'")

    def solve(ts, y0, args):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            0.1,
            y0,
            args=args,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    def solveExtrap(ts, y0, args):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            0.1,
            y0,
            args=args,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key, train_key, sample_key = jr.split(key, 5)

    # get the data
    ts, ys = get_data(
        batch_size, key=data_key, func=func, t_end=t_final, n_points=n_points
    )

    def add_gaussian_noise(ys, key, noise_level=0.25):
        noise = jr.normal(key, ys.shape) * noise_level
        return ys + noise

    # add some jitter to the data
    key_noise = jr.split(train_key, ys.shape[0])
    ys = jax.vmap(add_gaussian_noise)(ys, key_noise)

    def test_error(model, ts, ys, key):
        # calculate MSE error
        key_test = jr.split(key, ys.shape[0])
        latent, _, _ = jax.vmap(model._latent)(ts, ys, key=key_test)
        sample_y = jax.vmap(model._sample)(ts, latent)
        sample_y = np.asarray(sample_y)
        mse_ = (ys - sample_y) ** 2
        mse_ = jnp.sum(mse_, axis=1)
        mse = jnp.sum(mse_)
        return mse / ys.shape[0]

    def extrapolation_error(model, n_samples, param_bounds, key, t_ext=50):
        # calculate MSE error for extrapolated times
        e_key = jr.split(key, n_samples)
        args = tuple(
            jax.random.uniform(key, shape=(n_samples,), minval=lb, maxval=ub)
            for (lb, ub) in param_bounds
        )
        args = jnp.squeeze(jnp.asarray(args)).T
        t0 = jnp.zeros((n_samples,))
        t_e = t_ext * jnp.ones((n_samples,))
        make_ts = lambda t0, t_e: jnp.linspace(t0, t_e, 300)
        sample_t = jax.vmap(make_ts)(t0, t_e)
        ICs = jax.vmap(model.sample)(sample_t, key=e_key)[:, 0, :]
        exact_ys = jax.vmap(solveExtrap)(sample_t, ICs, args)
        latent, _, _ = jax.vmap(model._latent)(sample_t, exact_ys, key=e_key)
        sample_y = jax.vmap(model._sample)(sample_t, latent)
        mse_ = (exact_ys - sample_y) ** 2
        mse_ = jnp.sum(mse_, axis=1)
        mse = jnp.sum(mse_)
        return mse / n_samples

    def model_error(model, n_samples, param_bounds, key, t_ext=50):
        # calculate MSE error for extrapolated times
        e_key = jr.split(key, n_samples)
        #args = tuple(
        #    jax.random.uniform(key, shape=(n_samples,), minval=lb, maxval=ub)
        #    for (lb, ub) in param_bounds
        #)
        #args = jnp.squeeze(jnp.asarray(args)).T
        bounds = param_bounds
        a_key, b_key, c_key, d_key = jr.split(key, 4)
        # randomly sample for each value in the bounds 
        alpha = jax.random.uniform(a_key, shape=(n_samples,), minval=bounds[0][0], maxval=bounds[0][1])
        beta = jax.random.uniform(b_key, shape=(n_samples,), minval=bounds[1][0], maxval=bounds[1][1])
        delta = jax.random.uniform(c_key, shape=(n_samples,), minval=bounds[2][0], maxval=bounds[2][1])
        gamma = jax.random.uniform(d_key, shape=(n_samples,), minval=bounds[3][0], maxval=bounds[3][1])
        args = jnp.squeeze(jnp.asarray([alpha, beta, delta, gamma])).T

        t0 = jnp.zeros((n_samples,))
        t_e = t_ext * jnp.ones((n_samples,))
        t_ei = 10 * jnp.ones((n_samples,))
        make_ts = lambda t0, t_e: jnp.linspace(t0, t_e, t_ext * 3)
        sample_t = jax.vmap(make_ts)(t0, t_e)
        make_tsi = lambda t0, t_ei: jnp.linspace(t0, t_ei, 30)
        sample_ti = jax.vmap(make_tsi)(t0, t_ei)
        ICs = jr.uniform(
              key, (n_samples, 2), minval=2, maxval=5
        )  # ranomize the ICs
        exact_ys = jax.vmap(solveExtrap)(sample_t, ICs, args)
        exact_ysi = jax.vmap(solveExtrap)(sample_ti, ICs, args)
        exact_ysi = jax.vmap(add_gaussian_noise)(exact_ysi, e_key)
        latent, _, _ = jax.vmap(model._latent)(sample_ti, exact_ysi, key=e_key)
        sample_y = jax.vmap(model._sample)(sample_t, latent)
        mse_ = (exact_ys - sample_y) ** 2
        mse_ = jnp.sum(mse_, axis=1) / n_samples
        mse = jnp.sum(mse_)
        std = jnp.std(mse_)
        return mse, std
    # instantiate the model
    model = LatentODE(
        data_size=ys.shape[-1],
        hidden_size=hidden_size,
        latent_size=latent_size,
        width_size=width_size,
        depth=depth,
        key=model_key,
        alpha=alpha,
        lossType="mahalanobis",
    )

    # the bad model
    model2 = LatentODE(
        data_size=dataset_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        width_size=width_size,
        depth=depth,
        key=model_key,
        alpha=alpha,
        lossType="default",
    )

    modelName = MODEL_NAME
    model = eqx.tree_deserialise_leaves(modelName, model)

    modelName = MODEL_NAME2
    model2 = eqx.tree_deserialise_leaves(modelName, model)

    # get errors and std 
    bounds = [(1.0, 2.0), (1.0, 2.0), (0.5, 0.8), (0.5, 0.8)]
    mse_default50, std_default50 = model_error(model2, 100, [(1.0, 2.0), (1.0, 2.0), (0.5, 0.8), (0.5, 0.8)], key, t_ext=20)
    mse_distance50, std_distance50 = model_error(model, 100, [(1.0, 2.0), (1.0, 2.0), (0.5, 0.8), (0.5, 0.8)], key, t_ext=20)
    mse_default150, std_default150 = model_error(model2, 100, [(1.0, 2.0), (1.0, 2.0), (0.5, 0.8), (0.5, 0.8)], key, t_ext=90)
    mse_distance150, std_distance150 = model_error(model, 100, [(1.0, 2.0), (1.0, 2.0), (0.5, 0.8), (0.5, 0.8)], key, t_ext=90)
    print(f"Default model MSE: {mse_default50}")
    print(f"Std Default model: {std_default50}")
    print(f"Default model MSE150: {mse_default150}")
    print(f"Std Default model150: {std_default150}")
    print(f"Distance model MSE: {mse_distance50}")
    print(f"Std distance model MSE: {std_distance50}")
    print(f"Distance model MSE: {mse_distance150}")
    print(f"Std Distance model MSE: {std_distance150}")

    # fig, axs = plt.subplots(1, 2, figsize=(17, 2))
    # plt.subplots_adjust(wspace=0.25)

    # colour pallete
    c1 = "navy"  # line 1
    c2 = "cornflowerblue"  # line 2
    c3 = "darkorchid"  # shading 1
    c4 = "black"  # shading 2

    # create some sample times
    t_end = 90
    ext = 30  # Change this back 2 * t_final
    gap_start = 100
    gap_end = 200
    sample_t = jnp.linspace(0, t_end, 900)
    sample_t_plot = jnp.linspace(0, t_end, 900)
    sample_t_noisy = jnp.linspace(0, 15, 250)
    # randomly sample for ICs
    ICs = model.sample(sample_t, key=sample_key)[0, :]
    # ----------------- the first trajectory ----------------- #
    ICs = jnp.array([3, 3])
    args = (1.5, 1.5, 0.5, 0.5)
    # Generate the exact solution
    exact_y = solve(sample_t_plot, ICs, args)
    exact_y2 = solve(sample_t, ICs, args)
    exact_y_noisy = solve(sample_t_noisy, ICs, args)
    #exact_y_noisy = add_gaussian_noise(exact_y_noisy, key=sample_key, noise_level=0.02)

    # Get the latent mapping for the exact ODE
    latent, _, _ = model._latent(sample_t_noisy, exact_y_noisy, key=sample_key)
    latent2, _, _ = model2._latent(sample_t_noisy, exact_y_noisy, key=sample_key)
    # Get the predicted trajectory
    sample_y = model._sample(sample_t_plot, latent)
    sample_y2 = model2._sample(sample_t_plot, latent2)
    sample_key = jr.split(sample_key, 10)[1]
    # Now to plot the latent space trajectories
    sample_latent = model._sampleLatent(sample_t, latent)
    sample_latent2 = model2._sampleLatent(sample_t, latent2)
    # plot params
    sz = 4
    f_sz = 21

    # ----------------- the second trajectory ----------------- #
    ICs = jnp.array([3, 3])
    args = (1, 1, 0.75, 0.75)
    # Generate the exact solution
    exact_y_t2 = solve(sample_t_plot, ICs, args)
    exact_y2_t2 = solve(sample_t, ICs, args)
    exact_y_noisy_t2 = solve(sample_t_noisy, ICs, args)
    #exact_y_noisy = add_gaussian_noise(exact_y_noisy, key=sample_key, noise_level=0.02)

    # Get the latent mapping for the exact ODE
    latent_t2, _, _ = model._latent(sample_t_noisy, exact_y_noisy_t2, key=sample_key)
    latent2_t2, _, _ = model2._latent(sample_t_noisy, exact_y_noisy_t2, key=sample_key)
    # Get the predicted trajectory
    sample_y_t2 = model._sample(sample_t_plot, latent_t2)
    sample_y2_t2 = model2._sample(sample_t_plot, latent2_t2)

    # ---------------------------- The plot ---------------------------- #
    # --------------------------- first plot --------------------------- # 
    # Create figure and grid layout
    fig = plt.figure(figsize=(10, 4))
    gs = GridSpec(2, 2, width_ratios=[2, 1])
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    # plot the trajectories in data space
    ax = plt.subplot(gs[0, 0])
    ax.plot(sample_t, sample_y[:, 0], color=c1, lw=1, label=LAB_X, zorder=6)
    ax.plot(sample_t, sample_y[:, 1], color=c2, lw=1, label=LAB_Y, zorder=6)
    ax.plot(
        sample_t_plot, exact_y[:, 0], color="black", lw=0.5, zorder=5, label="exact"
    )
    ax.plot(sample_t_plot, exact_y[:, 1], color="black", lw=0.5, zorder=5)
    # ax.set_xlabel("time (s)", fontsize=f_sz)
    ax.axvspan(ext, t_end + 2, alpha=0.25, color=c3, zorder=1, label="extrapolation")
    ax.axvspan(
        0, sample_t_noisy[-1], alpha=0.25, color=c4, zorder=1, label="input data"
    )
    ax.set_xlim([0, t_end])
    ax.set_ylabel("population", fontsize=f_sz)
    # ax.set_ylim([-4,4])
    ax.legend(loc="upper right", fontsize=9, ncols=3,framealpha=1.0,
              bbox_to_anchor=(0.892, 1.35), fancybox=True).set_zorder(12)
    # ax.set_title("reconstruction", fontsize=f_sz)
    props = dict(boxstyle='square', facecolor='wheat', alpha=0.99)
    ax.text(0.9285, 0.89, 'trial 1',
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax.transAxes,
            fontsize=15,
            bbox=props).set_zorder(13)

    # plot the trajectories in data space
    ax = plt.subplot(gs[1, 0])
    ax.plot(sample_t, sample_y_t2[:, 0], color=c1, lw=1, label=LAB_X, zorder=6)
    ax.plot(sample_t, sample_y_t2[:, 1], color=c2, lw=1, label=LAB_Y, zorder=6)
    ax.plot(
        sample_t_plot, exact_y_t2[:, 0], color="black", lw=0.5, zorder=5, label="exact"
    )
    ax.plot(sample_t_plot, exact_y_t2[:, 1], color="black", lw=0.5, zorder=5)
    ax.set_xlabel("time (s)", fontsize=f_sz)
    ax.axvspan(ext, t_end + 2, alpha=0.25, color=c3, zorder=1, label="extrapolation")
    ax.axvspan(
        0, sample_t_noisy[-1], alpha=0.25, color=c4, zorder=1, label="input data"
    )
    ax.set_xlim([0, t_end])
    ax.set_ylabel("population", fontsize=f_sz)
    ax.text(0.9285, 0.89, 'trial 2',
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax.transAxes,
            fontsize=15,
            bbox=props).set_zorder(13)
    # the phase space plot
    ax = plt.subplot(gs[:, 1])
    sample_y_in = sample_y[sample_t < ext]
    sample_y_out = sample_y[sample_t >= ext]
    exact_y_in = exact_y[sample_t_plot < ext]
    exact_y_out = exact_y[sample_t_plot >= ext]
    ax.plot(
        sample_y[:, 0],
        sample_y[:, 1],
        color="firebrick",
        label="trial 1",
        alpha=0.5,
        lw=0.5
    )
    ax.scatter(
        exact_y[:, 0],
        exact_y[:, 1],
        color="firebrick",
        s=2,
        label="exact",
    )
    ax.set_xlabel(LAB_X, fontsize=f_sz)
    ax.plot(
        sample_y_t2[:, 0],
        sample_y_t2[:, 1],
        color="navy",
        label="trial 2",
        alpha=0.5,
        lw=0.5,
    )
    ax.scatter(
        exact_y_t2[:, 0],
        exact_y_t2[:, 1],
        color="navy",
        s=2,
    )
    ax.set_ylabel(LAB_Y, fontsize=f_sz)
    ax.legend()
    # ax.set_title("phase diagram", fontsize=f_sz)
    figname ="sample_lve_distance.png"
    plt.savefig(figname, bbox_inches="tight", dpi=200)
    figname2 = figname.replace(".png", ".pdf")
    plt.savefig(figname2, bbox_inches="tight", dpi=200)


    # --------------------------- The plot ---------------------------- #
    # -------------------------- second plot -------------------------- #
    # Create figure and grid layout
    fig = plt.figure(figsize=(10, 4))
    gs = GridSpec(2, 2, width_ratios=[2, 1])
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    # plot the trajectories in data space
    ax = plt.subplot(gs[0, 0])
    ax.plot(sample_t, sample_y2[:, 0], color=c1, lw=1, label=LAB_X, zorder=6)
    ax.plot(sample_t, sample_y2[:, 1], color=c2, lw=1, label=LAB_Y, zorder=6)
    ax.plot(
        sample_t_plot, exact_y2[:, 0], color="black", lw=0.5, zorder=5, label="exact"
    )
    ax.plot(sample_t_plot, exact_y2[:, 1], color="black", lw=0.5, zorder=5)
    # ax.set_xlabel("time (s)", fontsize=f_sz)
    ax.axvspan(ext, t_end + 2, alpha=0.25, color=c3, zorder=1, label="extrapolation")
    ax.axvspan(
        0, sample_t_noisy[-1], alpha=0.25, color=c4, zorder=1, label="input data"
    )
    ax.set_xlim([0, t_end])
    ax.set_ylabel("population", fontsize=f_sz)
    # ax.set_ylim([-4,4])
    ax.legend(loc="upper right", fontsize=9, ncols=3,framealpha=1.0,
              bbox_to_anchor=(0.892, 1.35), fancybox=True).set_zorder(12)
    # ax.set_title("reconstruction", fontsize=f_sz)
    props = dict(boxstyle='square', facecolor='wheat', alpha=0.99)
    ax.text(0.9285, 0.89, 'trial 1',
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax.transAxes,
            fontsize=15,
            bbox=props).set_zorder(13)

    # plot the trajectories in data space
    ax = plt.subplot(gs[1, 0])
    ax.plot(sample_t, sample_y2_t2[:, 0], color=c1, lw=1, label=LAB_X, zorder=6)
    ax.plot(sample_t, sample_y2_t2[:, 1], color=c2, lw=1, label=LAB_Y, zorder=6)
    ax.plot(
        sample_t_plot, exact_y2_t2[:, 0], color="black", lw=0.5, zorder=5, label="exact"
    )
    ax.plot(sample_t_plot, exact_y2_t2[:, 1], color="black", lw=0.5, zorder=5)
    ax.set_xlabel("time (s)", fontsize=f_sz)
    ax.axvspan(ext, t_end + 2, alpha=0.25, color=c3, zorder=1, label="extrapolation")
    ax.axvspan(
        0, sample_t_noisy[-1], alpha=0.25, color=c4, zorder=1, label="input data"
    )
    ax.set_xlim([0, t_end])
    ax.set_ylabel("population", fontsize=f_sz)
    ax.text(0.9285, 0.89, 'trial 2',
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax.transAxes,
            fontsize=15,
            bbox=props).set_zorder(13)
    # the phase space plot
    ax = plt.subplot(gs[:, 1])
    sample_y_in = sample_y2[sample_t < ext]
    sample_y_out = sample_y2[sample_t >= ext]
    exact_y_in = exact_y[sample_t_plot < ext]
    exact_y_out = exact_y[sample_t_plot >= ext]
    ax.plot(
        sample_y2[:, 0],
        sample_y2[:, 1],
        color="firebrick",
        label="trial 1",
        alpha=0.5,
        lw=0.5
    )
    ax.scatter(
        exact_y[:, 0],
        exact_y[:, 1],
        color="firebrick",
        s=2,
        label="exact",
    )
    ax.set_xlabel(LAB_X, fontsize=f_sz)
    ax.plot(
        sample_y2_t2[:, 0],
        sample_y2_t2[:, 1],
        color="navy",
        label="trial 2",
        alpha=0.5,
        lw=0.5,
    )
    ax.scatter(
        exact_y_t2[:, 0],
        exact_y_t2[:, 1],
        color="navy",
        s=2,
    )
    ax.set_ylabel(LAB_Y, fontsize=f_sz)
    ax.legend()
    # ax.set_title("phase diagram", fontsize=f_sz)
    figname = "sample_lve_baseline.png"
    plt.savefig(figname, bbox_inches="tight", dpi=200)
    figname2 = figname.replace(".png", ".pdf")
    plt.savefig(figname2, bbox_inches="tight", dpi=200)


    # IC predictor
    sample_t = jnp.linspace(0, t_end, 300)
    sample_t_noisy = jnp.linspace(1, 60, 200)
    # randomly sample for ICs
    ICs = jnp.array([3, 1])

    # Make a for loop around the IC prediction
    default_vec = []
    distance_vec = []
    time_vec = []
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for i in range(20):
        # generate big vectors of data
        IC_min = 1
        IC_max = 6
        trial_num = 5
        n_points = 120
        ykey, tkey1, tkey2 = jr.split(sample_key, 3)
        multi_key = jr.split(sample_key, trial_num)
        y0 = jr.uniform(
            ykey, (trial_num, 2), minval=IC_min, maxval=IC_max
        )  # ranomize the ICs
        y0 = jnp.array([3, 3])
        y0 = (jnp.expand_dims(y0, axis=0),) * trial_num
        y0 = jnp.concatenate(y0, axis=0)
        # Generate the exact solution
        # ts = jr.uniform(tkey1, (trial_num, n_points), minval=2, maxval=60)
        # ts = jnp.sort(ts)
        points = 3 * i + 80  # n_points
        ts = jnp.linspace(0, 60, points)
        ts = (jnp.expand_dims(ts, axis=0),) * trial_num
        ts = jnp.concatenate(ts, axis=0)

        # now make the time vectors for the reconstruction
        ts_r = jnp.linspace(0, 50, 300)
        ts_r = (jnp.expand_dims(ts_r, axis=0),) * trial_num
        ts_r = jnp.concatenate(ts_r, axis=0)

        # random args
        param_bounds = [(0.5, 1.5), (0.5, 1.5), (1.5, 2.5), (0.5, 1.5)]
        args = tuple(
            jax.random.uniform(key, shape=(trial_num,), minval=lb, maxval=ub)
            for (lb, ub) in param_bounds
        )
        args = jnp.squeeze(jnp.asarray(args)).T

        # args = jnp.array([1,1,2,1])
        # args = (jnp.expand_dims(args, axis=0),) * trial_num
        # args = jnp.concatenate(args, axis=0)
        y_noisy = jax.vmap(solve)(ts, y0, args)
        # y_noisy = jax.vmap(add_gaussian_noise)(y_noisy, key=multi_key)
        start_idx = int(2 * i)
        # now remove the values at start
        # ts = ts[:,start_idx:points]
        # y_noisy = y_noisy[:,start_idx:points, :]

        latent_m, _, _ = jax.vmap(model._latent)(ts, y_noisy, key=multi_key)
        latent2_m, _, _ = jax.vmap(model2._latent)(ts, y_noisy, key=multi_key)
        multi_IC_distance = jax.vmap(model._sample)(ts_r, latent_m)
        multi_IC_default = jax.vmap(model2._sample)(ts_r, latent2_m)

        # plot the iteritively improving trajectories
        ax[0].plot(
            ts_r[0:3, :],
            multi_IC_distance[0:3, :, 0],
            c="firebrick",
            alpha=0.03,
            zorder=1,
        )
        ax[0].plot(
            ts_r[0:3, :], multi_IC_distance[0:3, :, 1], c="black", alpha=0.03, zorder=1
        )

        ax[1].plot(
            ts_r[0:3, :],
            multi_IC_default[0:3, :, 0],
            c="firebrick",
            alpha=0.03,
            zorder=1,
        )
        ax[1].plot(
            ts_r[0:3, :], multi_IC_default[0:3, :, 1], c="black", alpha=0.03, zorder=1
        )

        print(f"IC default is {multi_IC_default[0:5,0,:]}")
        print(f"IC distance is {multi_IC_distance[0:5,0,:]}")

        default_error = jnp.sum((multi_IC_default[:, 0, :] - y0) ** 2) / trial_num
        distance_error = jnp.sum((multi_IC_distance[:, 0, :] - y0) ** 2) / trial_num

        default_vec.append(default_error)
        distance_vec.append(distance_error)
        time_vec.append(i + 2)

        print(f"default error is {default_error}")
        print(f"distance error is {distance_error}")

    # now plot the exact solutions in orange
    ax[0].scatter(
        ts[0, :], y_noisy[0, :, 0], s=10, c="firebrick", label="pred exact", zorder=10
    )
    ax[0].scatter(
        ts[0, :], y_noisy[0, :, 1], s=10, c="black", label="prey exact", zorder=10
    )
    ax[1].scatter(
        ts[0, :], y_noisy[0, :, 0], s=10, c="firebrick", label="prey_exact", zorder=10
    )
    ax[1].scatter(ts[0, :], y_noisy[0, :, 1], s=10, c="black", zorder=10)
    ax[0].set_xlabel("time (s)", fontsize=f_sz)
    ax[0].set_ylabel(r" population", fontsize=f_sz)
    ax[1].set_xlabel("time (s)", fontsize=f_sz)
    ax[0].legend()
    # add the title
    ax[0].set_title("minimum path loss", fontsize=f_sz)
    ax[1].set_title("default", fontsize=f_sz)

    plt.savefig("samples_lve.pdf", bbox_inches="tight", dpi=200)

    # plot the IC prediction
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot(time_vec, default_vec, c="darkorchid")
    ax.plot(time_vec, distance_vec, c="black")
    ax.scatter(time_vec, default_vec, label="standard loss", c="darkorchid")
    ax.scatter(time_vec, distance_vec, label="minimum path loss", c="black")
    ax.set_xlabel("number of data points", fontsize=f_sz)
    ax.set_ylabel(r"$\langle$ predicted IC MSE $\rangle$", fontsize=f_sz)
    ax.legend(fontsize=18)
    plt.savefig("_old_IC_predictor.pdf", bbox_inches="tight", dpi=200)

    # make a Umap
    import umap

    metric = "euclidean"  # {"mahalanobis", "wminkowski"}
    reducer = umap.UMAP(n_neighbors=50, metric=metric)
    reducer2 = umap.UMAP(n_neighbors=50, metric="mahalanobis")
    from sklearn.preprocessing import StandardScaler

    # making the data
    sample_t = jnp.linspace(0, 90, 500)
    ICs = jnp.array([3, 3])
    IC_vec = [
        jnp.array([5, 2]),
        jnp.array([4, 2]),
        jnp.array([4, 3]),
        jnp.array([5, 5]),
        jnp.array([3, 3]),
        jnp.array([2, 2]),
        jnp.array([3, 4]),
        jnp.array([2, 5]),
    ]
    # jnp.array([3,4]), jnp.array([2.5,4]), jnp.array([2,4]), jnp.array([1.5,4]),jnp.array([1, 4]) ]
    #fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    cmap_vec = ["Greys_r", "Blues_r", "RdPu_r", "Greens_r", "Reds_r", "twilight"]
    args_vec = [
        (1.0, 1.0, 0.5, 0.8),
        (1.5, 1.5, 0.8, 0.5),
        (2.0, 1.5, 0.5, 0.5),
        (1.5, 1.2, 0.75, 0.65),
        (1.75, 1.15, 0.75, 0.5),
        (1.2, 2.0, 0.9, 0.62),
    ]
    col_vec = ["black", "blue", "violet", "darkgreen", "firebrick", "orange"]
    sz = 50
    sz2 = 5
    for i in range(len(args_vec)):
        #ICs = IC_vec[i]
        args = args_vec[i]
        col = col_vec[i]
        cmap = cmap_vec[i]
        exact_y = solve(sample_t, ICs, args)
        latent, _, _ = model._latent(sample_t, exact_y, key=sample_key)
        latent2, _, _ = model2._latent(sample_t, exact_y, key=sample_key)
        sample_latent = model._sampleLatent(sample_t, latent)
        sample_latent2 = model2._sampleLatent(sample_t, latent2)

        # for UMAP plot
        #embedding2 = reducer.fit_transform(sample_latent)
        #embedding = reducer.fit_transform(sample_latent2)
        #embedding4 = reducer2.fit_transform(sample_latent)
        #embedding3 = reducer2.fit_transform(sample_latent2)

        #embedding = reducer.fit_transform(latent2)
        #embedding2 = reducer.fit_transform(latent)
        #embedding4 = reducer2.fit_transform(latent)
        #embedding3 = reducer2.fit_transform(latent2)

        # for Latent variable plot
        embedding2 = sample_latent[:, 0:2]
        embedding = sample_latent2[:, 0:2]
        embedding4 = sample_latent[:, 2:4]
        embedding3 = sample_latent2[:, 2:4]

        # make plots
        axs[0][0].scatter(
            embedding[:, 0], embedding[:, 1], c=sample_t, s=sz2, cmap=cmap, zorder=0
        )  # , label=f"IC {ICs}")
        axs[0][1].scatter(
            embedding2[:, 0], embedding2[:, 1], c=sample_t, s=sz2, cmap=cmap, zorder=0
        )  # , label=f"IC {ICs}")
        # add a colourbar labelled time if first plot
        if i == 0:
            cbar = plt.colorbar(
                axs[0][0].scatter(
                    embedding[:, 0], embedding[:, 1], c=sample_t, s=sz2, cmap=cmap
                )
            )
            cbar.set_label("time (s)", fontsize=f_sz)
            # now for the second plot
            cbar = plt.colorbar(
                axs[0][1].scatter(
                    embedding2[:, 0], embedding2[:, 1], c=sample_t, s=sz2, cmap=cmap
                )
            )
            cbar.set_label("time (s)", fontsize=f_sz)

        axs[0][0].scatter(
            embedding[0, 0],
            embedding[0, 1],
            c=col,
            edgecolor="black",
            s=sz,
            label=f"params {args}",
            zorder=10,
        )
        axs[0][1].scatter(
            embedding2[0, 0],
            embedding2[0, 1],
            edgecolor="black",
            c=col,
            s=sz,
            label=f"params {args}",
            zorder=10,
        )

        axs[0][0].scatter(
            embedding[-1, 0],
            embedding[-1, 1],
            c=col,
            marker="x",
            s=sz,
            zorder=10,
        )
        axs[0][1].scatter(
            embedding2[-1, 0],
            embedding2[-1, 1],
            c=col,
            marker="x",
            s=sz,
            zorder=10,
        )


        # --- now the second plot --- #

        # make plots
        axs[1][0].scatter(
            embedding3[:, 0], embedding3[:, 1], c=sample_t, s=sz2, cmap=cmap, zorder=0
        )  # , label=f"IC {ICs}")
        axs[1][1].scatter(
            embedding4[:, 0], embedding4[:, 1], c=sample_t, s=sz2, cmap=cmap, zorder=0
        )  # , label=f"IC {ICs}")
        # add a colourbar labelled time if first plot
        if i == 0:
            cbar = plt.colorbar(
                axs[1][0].scatter(
                    embedding3[:, 0], embedding3[:, 1], c=sample_t, s=sz2, cmap=cmap
                )
            )
            cbar.set_label("time (s)", fontsize=f_sz)
            # now for the second plot
            cbar = plt.colorbar(
                axs[1][1].scatter(
                    embedding4[:, 0], embedding4[:, 1], c=sample_t, s=sz2, cmap=cmap
                )
            )
            cbar.set_label("time (s)", fontsize=f_sz)

        axs[1][0].scatter(
            embedding3[0, 0],
            embedding3[0, 1],
            c=col,
            edgecolor="black",
            s=sz,
            label=f"params {args}",
            zorder=10,
        )
        axs[1][1].scatter(
            embedding4[0, 0],
            embedding4[0, 1],
            edgecolor="black",
            c=col,
            s=sz,
            label=f"params {args}",
            zorder=10,
        )

        axs[1][0].scatter(
            embedding3[-1, 0],
            embedding3[-1, 1],
            c=col,
            marker="x",
            s=sz,
            zorder=10,
            )

        axs[1][1].scatter(
            embedding4[-1, 0],
            embedding4[-1, 1],
            c=col,
            marker="x",
            s=sz,
            zorder=10,
        )

    axs[0][0].set_title("standard loss", fontsize=f_sz)
    axs[0][0].set_ylabel("metric: euclidean", fontsize=f_sz)
    axs[1][0].set_ylabel("metric: mahalanobis", fontsize=f_sz)
    axs[0][0].set_ylabel("latent 1", fontsize=f_sz)
    axs[0][0].set_xlabel("latent 2", fontsize=f_sz)
    axs[0][1].set_xlabel("latent 2", fontsize=f_sz)
    axs[1][0].set_ylabel("latent 3", fontsize=f_sz)
    axs[1][0].set_xlabel("latent 4", fontsize=f_sz)
    axs[1][1].set_xlabel("latent 4", fontsize=f_sz)
    axs[0][1].set_title("minimum path loss", fontsize=f_sz)
    axs[1][0].legend(
        loc="upper center",
        bbox_to_anchor=(1.3, -0.15),
        fancybox=True,
        shadow=True,
        ncol=3,
    )
    # axs[1].legend()
    plt.savefig("distance_map.pdf", bbox_inches="tight", dpi=200)

    # make a 3d umap
    reducer = umap.UMAP(n_neighbors=3, n_components=3)
    embedding = reducer.fit_transform(sample_latent)
    embedding2 = reducer.fit_transform(sample_latent2)
    # make plots
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        embedding[:, 2],
        c=sample_t,
        s=30,
        cmap="inferno",
    )
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(
        embedding2[:, 0],
        embedding2[:, 1],
        embedding2[:, 2],
        c=sample_t,
        s=30,
        cmap="inferno",
    )
    plt.savefig("umap3d.pdf", bbox_inches="tight", dpi=200)


# run the code son
main(
    train=False,
    dataset_size=22000,  # number of data n_points
    batch_size=256,  # batch size
    n_points=30,  # number of points in the ODE data
    lr=1e-2,  # learning rate
    steps=2501,  # number of training steps
    plot_every=1250,  # plot every n steps
    save_every=1250,  # save the model every n steps
    error_every=50,  # calculate the error every n steps
    hidden_size=12,  # hidden size of the RNN
    latent_size=4,  # latent size of the autoencoder
    width_size=60,  # width of the ODE
    depth=3,  # depth of the ODE
    alpha=2,  # strength of the path penalty
    seed=1992,  # random seed
    t_final=15,  # final time of the ODE (note this is randomised between t_final and 2*t_final)
    lossType="default",  # {default, mahalanobis, distance}
    func="LVE",  # {LVE, SHO, PFHO} Lotka-Volterra, Simple (damped) Harmonic Oscillator, Periodically Forced Harmonic Oscillator
    figname="lve_plot_test.png",  # name of the figure
    save_name="lve_test",  # name of the saved model
    MODEL_NAME="sequential_a2seq1_npoints_150_hsz12_lsz4_w60_d3_lossTypemahalanobis_step_5000.eqx" ,  # name of the model to load
    MODEL_NAME2="sequential_a2seq1_npoints_150_hsz12_lsz4_w60_d3_lossTypedefault_step_2000.eqx",  # name of the model to load
    #MODEL_NAME2="lve_new_a1__npoints_250_hsz5_lsz5_w60_d3_lossTypemahalanobis_step_10000.eqx",  # name of the model to load
    #MODEL_NAME="lve_a2__npoints_150_hsz4_lsz4_w60_d3_lossTypemahalanobis_step_8000.eqx",
    #MODEL_NAME2="lve_a1__npoints_150_hsz12_lsz5_w60_d3_lossTypedefault_step_12000.eqx"
)


# ---------------------------------------- #
# For Damped Harmonic Oscillator
# ---------------------------------------- #
# Hyperparams:
# hidden_size= 6
# latent_size= 2
# width_size= 16 # 20 for gap test
# depth= 2
# lr = 1e2
# theta = 0.12
# steps = 2501
# IC (1 -> 4) (1 -> 4)
# ---------------------------------------- #


# ---------------------------------------- #
# For Lotka-Volterra Equations
# see: https://arxiv.org/pdf/2105.03835.pdf
# ---------------------------------------- #
# Hyperparams:
# hidden_size= 16
# latent_size= 8
# width_size= 100
# depth= 3
# lr = 5e-3
# (0.5,1.5)  (0.5,1.5)  (1.5,2.5)  (0.5,1.5)
# ---------------------------------------- #
