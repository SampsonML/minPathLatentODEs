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

    @staticmethod
    def _loss(ys, pred_ys, mean, std):
        # -log p_θ with Gaussian p_θ
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
        # KL(N(mean, std^2) || N(0, 1))
        variational_loss = 0.5 * jnp.sum(mean**2 + std**2 - 2 * jnp.log(std) - 1)
        return reconstruction_loss + variational_loss

    @staticmethod
    def _pathpenaltyloss(self, ts, ys, pred_ys, pred_latent, mean, std, key):
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
        alpha = self.alpha  # 1 # weighting parameter for distance penalty
        return reconstruction_loss + variational_loss + alpha * d_latent

    @staticmethod
    def _distanceloss(self, ts, ys, pred_ys, pred_latent, mean, std, key):
        # TODO: implement this bypassing the need for the variational loss
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
        alpha = self.alpha  # 1 # weighting parameter for distance penalty
        return reconstruction_loss + alpha * d_latent

    # Run both encoder and decoder during training.
    def train(self, ts, ys, *, key):
        latent, mean, std = self._latent(ts, ys, key)
        pred_ys = self._sample(ts, latent)
        pred_latent = self._sampleLatent(ts, latent)
        # the classic VAE based LatentODE-RNN from https://arxiv.org/abs/1907.03907
        if self.lossType == "default":
            return self._loss(ys, pred_ys, mean, std)
        # the classic LatentODE-RNN with the path length penalty
        elif self.lossType == "mahalanobis":
            return self._pathpenaltyloss(
                self, ts, ys, pred_ys, pred_latent, mean, std, key
            )
        # our new autoencoder (not VAE) LatentODE-RNN with no variational loss TODO: implement this
        elif self.lossType == "distance":
            raise ValueError("lossType 'distance' not yet implemented")
        # return self._distanceloss(self, ts, ys, pred_ys, pred_latent, mean, std, key)
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
    y0 = jr.uniform(ykey, (dataset_size, 2), minval=2, maxval=5)  # ranomize the ICs
    t0 = 0
    # randomize the total time series between t_end and 2 * t_end (t_end is user defined)
    t1 = t_end + 1 * jr.uniform(tkey1, (dataset_size,), minval=0, maxval=t_end)
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
    # Periodically forced hamonic oscillator
    def PFHO(t, y, args):
        y1, y2 = y
        w, b, k, force = args
        dy1 = y2
        dy2 = force * jnp.cos(w * t) - b * y2 - k * y1
        d_y = jnp.array([dy1, dy2])
        return d_y

    PFHO_args = (1, 1, 1, 3)  # w, b, k, force

    if func == "LVE":
        vector_field = LVE
        args = LVE_args
        bounds = [
            (0.5, 1.5),
            (0.5, 1.5),
            (1.5, 2.5),
            (0.5, 1.5),
        ]  # same as https://arxiv.org/pdf/2105.03835.pdf
        key = jax.random.PRNGKey(0)
        #key_dataset = jr.split(key, dataset_size)
        args = tuple(
            jax.random.uniform(key, shape=(1,), minval=lb, maxval=ub)
            for (lb, ub) in bounds
        )
        args = jnp.squeeze(jnp.asarray(args))
    elif func == "SHO":
        vector_field = SHO
        args = SHO_args  # Fixed damping rate
    elif func == "PFHO":
        vector_field = PFHO
        args = PFHO_args  # fixed for now, not using this test
    else:
        raise ValueError("func must be one of 'LVE', 'SHO', 'PFHO'")

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


    def solveLVE(ts, y0, key):
        bounds = [
            (0.5, 1.5),
            (0.5, 1.5),
            (1.5, 2.5),
            (0.5, 1.5),
        ]  # same as https://arxiv.org/pdf/2105.03835.pdf
        args = tuple(jax.random.uniform(key, shape=(1,), minval=lb, maxval=ub) for (lb, ub) in bounds)
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
        return sol.ys

    # for now seperate the call to LVE to allow random input params
    if func == "LVE":
        key = jax.random.PRNGKey(0)
        key_dataset = jr.split(key, dataset_size)
        ys = jax.vmap(solveLVE)(ts, y0, key_dataset)
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
    dataset_size=20000,
    batch_size=256,
    n_points=100,
    lr=1e-2,
    steps=30,
    plot_every=10,
    save_every=10,
    hidden_size=8,
    latent_size=2,
    width_size=8,
    depth=2,
    alpha=1,
    seed=1992,
    t_final=20,
    lossType="default",
    func="PFHO",
    figname="latent_ODE.png",
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

    LVE_args = (1.5, 1.5, 2.5, 1.5)  # a=prey-growth, b, c, d

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
    # Periodically forced hamonic oscillator
    def PFHO(t, y, args):
        y1, y2 = y
        w, b, k, force = args
        dy1 = y2
        dy2 = force * jnp.cos(w * t) - b * y2 - k * y1
        d_y = jnp.array([dy1, dy2])
        return d_y

    PFHO_args = (1, 1, 1, 3)  # w, b, k, force

    def PO(t, y, args):
        dy = jnp.sin(t)
        return dy

    if func == "LVE":
        vector_field = LVE
        args = LVE_args
        rows = 3
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
    elif func == "PFHO":
        vector_field = PFHO
        args = PFHO_args
        rows = 3
        TITLE = "Latent ODE Model: Periodically Forced Harmonic Oscillator"
        LAB_X = "position"
        LAB_Y = "velocity"
    else:
        raise ValueError("func must be one of 'LVE', 'SHO', 'PFHO'")

    def solve(ts, y0):
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
        dataset_size, key=data_key, func=func, t_end=t_final, n_points=n_points
    )

    # instantiate the model
    model = LatentODE(
        data_size=ys.shape[-1],
        hidden_size=hidden_size,
        latent_size=latent_size,
        width_size=width_size,
        depth=depth,
        key=model_key,
        alpha=alpha,
        lossType=lossType,
    )

    @eqx.filter_value_and_grad
    def loss(model, ts_i, ys_i, key_i):
        batch_size, _ = ts_i.shape
        key_i = jr.split(key_i, batch_size)
        loss = jax.vmap(model.train)(ts_i, ys_i, key=key_i)
        return jnp.mean(loss)

    @eqx.filter_jit
    def make_step(model, opt_state, ts_i, ys_i, key_i):
        value, grads = loss(model, ts_i, ys_i, key_i)
        key_i = jr.split(key_i, 1)[0]
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return value, model, opt_state, key_i

    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # Plot results
    # num_plots = 1 + (steps - 1) // plot_every
    num_plots = (steps) // plot_every  # don't plot initial untrained model
    # if ((steps - 1) % plot_every) != 0:
    #    num_plots += 1
    fig, axs = plt.subplots(rows, num_plots, figsize=(num_plots * 4, rows * 4 - 2))

    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    idx = 0
    f_sz = 16
    loss_vector = []
    path_vector = []
    mse_vec = []
    for step, (ts_i, ys_i) in zip(
        range(steps), dataloader((ts, ys), batch_size, key=loader_key)
    ):
        start = time.time()
        value, model, opt_state, train_key = make_step(
            model,
            opt_state,
            ts_i,
            ys_i,
            train_key,
        )
        end = time.time()
        print(f"Step: {step}, Loss: {value}, Computation time: {end - start}")
        loss_vector.append(value)
        # Path length calculation
        key_path = jr.split(train_key, 1)[0]
        key_path = jr.split(key_path, ts_i.shape[0])
        path_len = jax.vmap(model.pathLength)(ts=ts_i, ys=ys_i, key=key_path)
        path_vector.append(jnp.mean(path_len))

        # calculate MSE extrapolation error
        t_end = 60
        sample_t = jnp.linspace(0, t_end, 300)
        ICs = model.sample(sample_t, key=sample_key)[0, :]  # randomly sample for ICs
        exact_y = solve(sample_t, ICs)
        latent, _, _ = model._latent(sample_t, exact_y, key=sample_key)
        sample_y = model._sample(sample_t, latent)
        # sample_y = model.sample(sample_t, key=sample_key)
        # sample_t = np.asarray(sample_t)
        sample_y = np.asarray(sample_y)
        # exact_y = solve(sample_t, sample_y[0, :])
        mse_ = (exact_y - sample_y) ** 2
        mse_ = jnp.sum(mse_, axis=1)
        mse_ = jnp.sum(mse_)
        mse_vec.append(mse_)

        # save the model
        SAVE_DIR = "saved_models"
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        if (step % save_every) == 0 or step == steps - 1:
            fn = SAVE_DIR + "/latentODE" + str(step) + ".eqx"
            eqx.tree_serialise_leaves(fn, model)

        # make the plot
        if ((step % plot_every) == 0 and (step > 0)) or step == steps - 1:
            # create some sample times
            t_end = 70
            ext = t_final + 10
            sample_t = jnp.linspace(0, t_end, 300)
            # latent_sample = model._latent(sample_t, ys[0], key=sample_key)
            # randomly sample for ICs
            # sample_y = model.sample(sample_t, key=sample_key)
            ICs = model.sample(sample_t, key=sample_key)[0, :]
            # Generate the exact solution
            exact_y = solve(sample_t, ICs)
            # Get the latent mapping for the exact ODE
            latent, _, _ = model._latent(sample_t, exact_y, key=sample_key)
            # Get the predicted trajectory
            sample_y = model._sample(sample_t, latent)
            # Now to plot the latent space trajectories
            sample_latent = model._sampleLatent(sample_t, latent)

            # Make arrays for numpy plotting convenience
            sample_latent = np.asarray(sample_latent)
            sample_t = np.asarray(sample_t)
            sample_y = np.asarray(sample_y)
            # exact_y = solve(sample_t, sample_y[0, :])
            sz = 2
            # plot the trajectories in data space
            ax = axs[0][idx]
            if idx == 0:
                ax.plot(sample_t, sample_y[:, 0], color="firebrick", label=LAB_X)
            if idx == 0:
                ax.plot(sample_t, sample_y[:, 1], color="steelblue", label=LAB_Y)
            if idx == 0:
                ax.scatter(-10, 2, color="black", s=sz, label="exact")
            ax.scatter(sample_t, exact_y[:, 0], color="firebrick", s=sz)
            ax.scatter(sample_t, exact_y[:, 1], color="steelblue", s=sz)
            ax.set_title(f"training step: {step}", fontsize=f_sz)
            ax.set_xlabel("time (s)", fontsize=f_sz)
            if idx == 0:
                ax.axvspan(ext, t_end + 2, alpha=0.2, color="coral")
            ax.set_xlim([0, t_end])
            if idx == 0:
                ax.set_ylabel("arb", fontsize=f_sz)
                ax.legend()
            else:
                ax.plot(sample_t, sample_y[:, 0], color="firebrick")
                ax.plot(sample_t, sample_y[:, 1], color="steelblue")
                ax.axvspan(
                    ext, t_end + 2, alpha=0.2, color="coral", label="extrapolation"
                )
                ax.legend()

            # the phase space plot
            ax = axs[1][idx]
            sample_y_in = sample_y[sample_t < ext]
            sample_y_out = sample_y[sample_t >= ext]
            exact_y_in = exact_y[sample_t < ext]
            exact_y_out = exact_y[sample_t >= ext]
            ax.plot(
                sample_y_in[:, 0],
                sample_y_in[:, 1],
                color="darkgray",
                label="LatentODE",
            )
            ax.scatter(
                exact_y_in[:, 0],
                exact_y_in[:, 1],
                color="darkgray",
                s=sz,
                label="exact",
            )
            ax.plot(
                sample_y_out[:, 0],
                sample_y_out[:, 1],
                color="coral",
                label="ODE: extrapolated",
            )
            ax.scatter(exact_y_out[:, 0], exact_y_out[:, 1], color="coral", s=sz)
            ax.set_xlabel(LAB_X, fontsize=f_sz)
            if idx == 0:
                ax.set_ylabel(LAB_Y, fontsize=f_sz)
                ax.legend()

            # now the latent space plot
            ax = axs[2][idx]
            cmap = plt.get_cmap("plasma")
            for i in range(sample_latent.shape[1]):
                name = f"latent{i}"
                color = cmap(i / sample_latent.shape[1])
                ax.plot(sample_t, sample_latent[:, i], color=color, label=name)
            ax.set_xlabel("time (s)", fontsize=f_sz)
            ax.axvspan(ext, t_end + 2, alpha=0.2, color="coral")
            ax.set_xlim([0, t_end])
            if idx == 0:
                ax.set_ylabel("arb", fontsize=f_sz)
                ax.legend()

            if rows > 3:
                ax = axs[3][idx]
                latent_in = sample_latent[sample_t < ext]
                latent_out = sample_latent[sample_t >= ext]
                ax.plot(
                    latent_in[:, 0],
                    latent_in[:, 1],
                    color="darkgray",
                    label="LatentODE",
                )
                ax.plot(
                    latent_out[:, 0],
                    latent_out[:, 1],
                    color="coral",
                    label="ODE: extrapolated",
                )
                ax.set_xlabel("latent 0", fontsize=f_sz)
                if idx == 0:
                    ax.set_ylabel("latent 1", fontsize=f_sz)
            idx += 1

    # plt.suptitle(TITLE, y=0.935, fontsize=20)
    plt.savefig(figname, bbox_inches="tight", dpi=200)
    figname2 = figname.replace(".png", ".pdf")
    plt.savefig(figname2, bbox_inches="tight", dpi=200)

    # Plot the loss figure and interpolation error
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    # the loss
    ax[0].plot(loss_vector, color="black")
    ax[0].set_xlabel("step", fontsize=f_sz)
    ax[0].set_ylabel("loss", fontsize=f_sz)

    # the interpolation error
    error = (sample_y - exact_y) ** 2
    error = np.sum(error, axis=1)
    ax[1].plot(sample_t, error, color="gray")
    ax[1].axvspan(ext, t_end + 2, alpha=0.2, color="coral")
    ax[1].set_xlabel("time", fontsize=f_sz)
    ax[1].set_ylabel("square error", fontsize=f_sz)
    ax[1].set_xlim([0, t_end])

    # rename and save the figure
    figname = figname.replace(".png", "_loss.png")
    plt.savefig(figname, bbox_inches="tight", dpi=200)
    figname2 = figname.replace(".png", ".pdf")
    plt.savefig(figname2, bbox_inches="tight", dpi=200)

    # Plot the loss figure and interpolation error
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    # the loss
    ax[0].plot(mse_vec[5:-1], color="black")
    ax[0].set_xlabel("step", fontsize=f_sz)
    ax[0].set_ylabel("MSE", fontsize=f_sz)

    # the interpolation error
    ax[1].plot(path_vector[5:-1], color="firebrick")
    ax[1].set_xlabel("step", fontsize=f_sz)
    ax[1].set_ylabel(r"$\langle \sum_i^{n-1} d_{M,i} \rangle$", fontsize=f_sz)

    # rename and save the figure
    figname = figname.replace(".png", "_path.png")
    plt.savefig(figname, bbox_inches="tight", dpi=200)
    figname2 = figname.replace(".png", ".pdf")
    plt.savefig(figname2, bbox_inches="tight", dpi=200)

    # save the vectors to compare
    filename = figname2.replace(".pdf", "_mse.npy")
    np.save(filename, mse_vec)
    filename = filename.replace("_mse.npy", "_path.npy")
    np.save(filename, path_vector)


# run the code son
main(
    dataset_size=20000,  # number of data n_points 
    batch_size=256,  # batch size
    n_points=150,  # number of points in the ODE data
    lr=1e-2,  # learning rate
    steps=301,  # number of training steps
    plot_every=100,  # plot every n steps
    save_every=100,  # save the model every n steps
    hidden_size=8,  # hidden size of the RNN
    latent_size=6,  # latent size of the autoencoder
    width_size=32,  # width of the ODE
    depth=3,  # depth of the ODE
    alpha=2.5,  # strength of the path penalty
    seed=1992,  # random seed
    t_final=10,  # final time of the ODE (note this is randomised between t_final and 2*t_final)
    lossType="mahalanobis",  # {default, mahalanobis, distance}
    func="LVE",  # {LVE, SHO, PFHO} Lotka-Volterra, Simple (damped) Harmonic Oscillator, Periodically Forced Harmonic Oscillator
    figname="LVE_varied_maha_dynamics.png",
)


# ---------------------------------------- #
# For Damped Harmonic Oscillator
# ---------------------------------------- #
# Hyperparams:
# hidden_size= 6
# latent_size= 2
# width_size= 16
# depth= 2
# lr = 1e2
# theta = 0.12
# steps = 2501
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
