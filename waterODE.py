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
        dt0 = 1.0  # selected as a reasonable choice for this problem
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
    IC_min = 5
    IC_max = 15
    y0 = jr.uniform(
        ykey, (dataset_size, 1), minval=IC_min, maxval=IC_max
    )  # ranomize the ICs
    t0 = 0
    # randomize the total time series between t_end and 2 * t_end (t_end is user defined)
    t1 = t_end + 1 * jr.uniform(tkey1, (dataset_size,), minval=1, maxval=5)
    ts = jr.uniform(tkey2, (dataset_size, n_points)) * (t1[:, None] - t0) + t0
    ts = jnp.sort(ts)
    dt0 = 0.1 # needed for single measurement per day


    # --------------------------------------
    # WaterBucket
    def Water(t, y, args):
        S = y 
        k, alpha, P = args 
        p = P * jnp.abs(jnp.sin(t))
        dy = p - k * S**alpha #  https://sml505.pmelchior.net/SimulationBasedInference.html#example-a-simple-reservoir
        return dy
    args = (0.5, 1.5, 0.5)  # k, alpha, P

    def solve(ts, y0):
        bounds = [
            (0.02, 1),
            (1.2, 2.8),
            (1, 5), # constant rain level for now
        ]  # bounds for water model
        args = tuple(
            jax.random.uniform(key, shape=(1,), minval=lb, maxval=ub)
            for (lb, ub) in bounds
        )
        args = jnp.squeeze(jnp.asarray(args))
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(Water),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            args=args,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys


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
    steps=1001,
    plot_every=250,
    save_every=250,
    error_every=20,
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
):

    # make random key
    key = jr.PRNGKey(seed)
    
    # Defining vector fields again for use in visualisation
    # --------------------------------------
    # WaterBucket
    
    def Water(t, y, args):
        S = y 
        k, alpha, P = args
        p = P * jnp.abs(jnp.sin(t))
        dy = p - k * S**alpha #  https://sml505.pmelchior.net/SimulationBasedInference.html#example-a-simple-reservoir
        return dy
    # rain is a Gaussian with a max height 1 of length n_points
    args = (0.5, 1.5, 2.5)  # k, alpha, P
    vector_field = Water 
    rows = 3 
    TITLE = "Latent ODE Model: Water Bucket"
    LAB_X = "time"
    LAB_Y = "water level"

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

    data_key, model_key, loader_key, train_key, sample_key = jr.split(key, 5)

    # get the data
    ts, ys = get_data(
        dataset_size, key=data_key, func=func, t_end=t_final, n_points=n_points
    )


    # make a test split -- randomly select 10% of the data for testing
    test_size = int(0.1 * dataset_size)
    test_idx = jr.choice(data_key, dataset_size, (test_size,), replace=False)
    train_idx = jnp.setdiff1d(jnp.arange(dataset_size), test_idx)
    ts_train, ys_train = ts[train_idx], ys[train_idx]
    ts_test, ys_test = ts[test_idx], ys[test_idx]


    # remove some inner data points so that the inner third of the data is gone
    def cut_mid(ts, ys):
        # indices = jnp.where((ts < 20) & (ts > 10))
        gap1 = int(len(ts) / 3.5)
        gap2 = int(gap1 * 2.5)
        ts = jnp.concatenate([ts[0:gap1], ts[gap2:]])
        ys = jnp.concatenate([ys[0:gap1], ys[gap2:]])
        return ts, ys

    #ts_train, ys_train = jax.vmap(cut_mid)(ts_train, ys_train)

    def add_gaussian_noise(ys, key, noise_level=0.2):
        noise = jr.normal(key, ys.shape) * noise_level
        return ys + noise

    # add some jitter to the data
    key_noise = jr.split(train_key, ys_train.shape[0])
    ys_train = jax.vmap(add_gaussian_noise)(ys_train, key_noise)

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
    num_plots = (steps) // plot_every  # don't plot initial untrained model
    fig, axs = plt.subplots(rows, num_plots, figsize=(num_plots * 4, rows * 4 - 2))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    idx = 0
    f_sz = 16
    loss_vector = []
    path_vector = []
    mse_vec = []
    ext_vec = []
    for step, (ts_i, ys_i) in zip(
        range(steps), dataloader((ts_train, ys_train), batch_size, key=loader_key)
    ):
        if train:
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

        # load the model instead here
        else:
            modelName = "saved_models/" + MODEL_NAME
            model = eqx.tree_deserialise_leaves(modelName, model)

        # NOTE: Just for one off visualisation purposes
        #if step ==0:
        #    fig2 = plt.figure(figsize=(10, 2))
        #    plt.subplots_adjust(wspace=0.2, hspace=0.2)
        #    plt.subplot(1,2,1)
        #    plt.plot(ts_i[0:250,:], ys_i[0:250,:], lw=1, c='navy', alpha=0.15, zorder=0)
            #plt.scatter(ts_i[30,:], ys_i[30,:], lw=0.1, c='darkorange',edgecolor='black',alpha=1, s=19, zorder=5)
            #plt.scatter(ts_i[0:250,:], ys_i[0:250,:,0], lw=1, c='black', alpha=0.10, s=4,zorder=0)
        #    plt.ylabel("pop (prey)", fontsize=16)
        #    plt.xlabel("time (s)", fontsize=16)

         #   plt.subplot(1,2,2)
         #   plt.plot(ts_i[0:250,:], ys_i[0:250,:,1], lw=1, c='navy', alpha=0.15,zorder=0)
            #plt.plot(ts_i[20,:], ys_i[20,:,1], lw=1, c='darkorange', alpha=0.15, zorder=0)
            #plt.scatter(ts_i[0:250,:], ys_i[0:250,:,1], lw=1, c='black', alpha=0.1, s=4,zorder=0)
          #  plt.scatter(ts_i[30,:], ys_i[30,:,1], lw=0.1, c='darkorange', edgecolor='black',alpha=1, s=19, zorder=5)
          #  plt.xlabel("time (s)", fontsize=16)
          #  plt.ylabel("pop (pred)", fontsize=16)
        #    plt.savefig("training_data.pdf", dpi=300, bbox_inches="tight")

        # track the path lengths and errors
        if step % error_every == 0:
            # path length calculation do this for the paths used in training
            key_path = jr.split(train_key, 1)[0]
            key_path = jr.split(key_path, ts_i.shape[0])
            path_len = jax.vmap(model.pathLength)(ts=ts_i, ys=ys_i, key=key_path)
            path_vector.append(jnp.mean(path_len))

            # calculate MSE error
            mse = test_error(model, ts_test, ys_test, key=sample_key)
            mse_vec.append(mse)

            # calculate extrapolation error
            n_samples = 256
            param_bounds = [(0.5, 0.5), (1.5, 1.5), (0.01, 0.01),] # for water
            ext = extrapolation_error(
                model, n_samples, param_bounds=param_bounds, key=sample_key, t_ext=60
            )
            ext_vec.append(ext)

        # save the model
        SAVE_DIR = "saved_models"
        save_suffix = (
            "_npoints_"
            + str(n_points)
            + "_hsz"
            + str(hidden_size)
            + "_lsz"
            + str(latent_size)
            + "_w"
            + str(width_size)
            + "_d"
            + str(depth)
            + "_lossType"
            + lossType
        )
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        if (step % save_every) == 0 or step == steps - 1:
            fn = SAVE_DIR + "/" + save_name + save_suffix + "_step_" + str(step) + ".eqx"
            eqx.tree_serialise_leaves(fn, model)

        # make the plot
        if ((step % plot_every) == 0 and (step > 0)) or step == steps - 1:
            # colour pallete
            c1 = "black"  # line 1
            c2 = "firebrick"  # line 2
            c3 = "coral"  # shading 1
            c4 = "black"  # shading 2

            # create some sample times
            t_end = 60
            ext = 20  # Change this back 2 * t_final
            gap_start = 100
            gap_end = 200
            sample_t = jnp.linspace(0, t_end, 300)
            sample_t_plot = jnp.linspace(0, t_end, 100)
            sample_t_noisy = jnp.linspace(0, t_final, 10)
            # randomly sample for ICs
            ICs = model.sample(sample_t, key=sample_key)[0]
            #print(ICs)
            ICs = jnp.array([10])
            # Generate the exact solution
            exact_y = solve(sample_t_plot, ICs)
            exact_y2 = solve(sample_t, ICs)
            exact_y_noisy = solve(sample_t_noisy, ICs)
            #exact_y_noisy = add_gaussian_noise(exact_y_noisy, key=sample_key, noise_level=0.3)
            # Get the latent mapping for the exact ODE
            #latent, _, _ = model._latent(sample_t, exact_y, key=sample_key)
            latent, _, _ = model._latent(sample_t_noisy, exact_y_noisy, key=sample_key)
            # Get the predicted trajectory
            sample_y = model._sample(sample_t, latent)
            # Now to plot the latent space trajectories
            sample_latent = model._sampleLatent(sample_t, latent)

            # Make arrays for numpy plotting convenience
            sample_latent = np.asarray(sample_latent)
            sample_t = np.asarray(sample_t)
            sample_y = np.asarray(sample_y)
            sz = 4
            # plot the trajectories in data space
            ax = axs[0][idx]
            if idx == 0:
                ax.plot(sample_t, sample_y, color=c2, label="model", zorder=6)
            if idx == 0:
                ax.scatter(-10, 2, color="black", s=sz, label="exact", zorder=5)
                ax.scatter(sample_t_noisy, exact_y_noisy, color=c1, marker='x' ,s=20, zorder=8, label='data')
            ax.scatter(sample_t_plot, exact_y, color=c1, s=sz, zorder=5)
            ax.set_title(f"training step: {step}", fontsize=f_sz)
            ax.set_xlabel("time (s)", fontsize=f_sz)
            if idx == 0:
                ax.axvspan(ext, t_end + 2, alpha=0.25, color=c3, zorder=1)
                ax.axvspan(gap_start, gap_end, alpha=0.25, color=c4, zorder=0)
            ax.set_xlim([0, t_end])
            if idx == 0:
                ax.set_ylabel("water level", fontsize=f_sz)
                ax.legend()
            else:
                ax.plot(sample_t, sample_y, color=c2, zorder=6)
                ax.axvspan(ext, t_end + 2, alpha=0.25, color=c3, label="extrapolation")
                ax.axvspan(
                    gap_start,
                    gap_end,
                    alpha=0.25,
                    color=c4,
                    zorder=0
                )
                ax.legend()


            # now the latent space plot
            ax = axs[2][idx]
            cmap = plt.get_cmap("plasma")
            for i in range(sample_latent.shape[1]):
                name = f"latent{i}"
                color = cmap(i / sample_latent.shape[1])
                ax.plot(sample_t, sample_latent[:, i], color=color, label=name)
            ax.set_xlabel("time (s)", fontsize=f_sz)
            ax.axvspan(ext, t_end + 2, alpha=0.25, color="coral")
            ax.set_xlim([0, t_end])
            if idx == 0:
                ax.set_ylabel("arb", fontsize=f_sz)
                ax.legend()


            ax = axs[1][idx]
            error = (sample_y - exact_y2) ** 2
            error = np.sum(error, axis=1)
            ax.plot(sample_t, error, color="black")
            ax.axvspan(ext, t_end + 2, alpha=0.25, color=c3, label="extrapolation")
            ax.axvspan(
                    gap_start,
                    gap_end,
                    alpha=0.25,
                    color=c4,
                    zorder=0,
                    label="missing data",
                    )
            ax.set_xlabel("time (s)", fontsize=f_sz)
            if idx == 0:
                ax.set_ylabel("MSE", fontsize=f_sz)
            ax.set_xlim([0, t_end])
            idx += 1

    # plt.suptitle(TITLE, y=0.935, fontsize=20)
    plt.savefig(figname, bbox_inches="tight", dpi=200)
    figname2 = figname.replace(".png", ".pdf")
    plt.savefig(figname2, bbox_inches="tight", dpi=200)

    if train:
        # Plot the loss figure and interpolation error
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))

        # the loss
        ax[0].plot(loss_vector, color="black")
        ax[0].set_xlabel("step", fontsize=f_sz)
        ax[0].set_ylabel("loss", fontsize=f_sz)

        # the interpolation error
        error = (sample_y - exact_y2) ** 2
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

        # Plot the interpolation and extrapolation error and path length
        fig, ax = plt.subplots(1, 3, figsize=(12, 3))

        # the interpolation error
        ax[0].plot(mse_vec[1:-1], color="black")
        ax[0].set_xlabel("step", fontsize=f_sz)
        ax[0].set_ylabel("MSE", fontsize=f_sz)

        # the extrapolation error
        ax[1].plot(ext_vec[1:-1], color="black")
        ax[1].set_xlabel("step", fontsize=f_sz)
        ax[1].set_ylabel("extrapolation error", fontsize=f_sz)

        # the path length
        ax[2].plot(path_vector[1:-1], color="firebrick")
        ax[2].set_xlabel("step", fontsize=f_sz)
        ax[2].set_ylabel(r"path length", fontsize=f_sz)

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
        filename = filename.replace("_path.npy", "_ext.npy")
        np.save(filename, ext_vec)

    # make U-maps of latent parameters


# run the code son
main(
    train=True,
    dataset_size=22000,  # number of data n_points
    batch_size=256,  # batch size
    n_points=50,  # number of points in the ODE data
    lr=1e-2,  # learning rate
    steps=31,  # number of training steps
    plot_every=10,  # plot every n steps
    save_every=10,  # save the model every n steps
    error_every=20,  # calculate the error every n steps
    hidden_size=8,  # hidden size of the RNN
    latent_size=3,  # latent size of the autoencoder
    width_size=24,  # width of the ODE
    depth=2,  # depth of the ODE
    alpha=3,  # strength of the path penalty
    seed=1992,  # random seed
    t_final=25,  # final time of the ODE (note this is randomised between t_final and 2*t_final)
    lossType="mahalanobis",  # {default, mahalanobis, distance}
    func="SHO",  # {LVE, SHO, PFHO} Lotka-Volterra, Simple (damped) Harmonic Oscillator, Periodically Forced Harmonic Oscillator
    figname="water_distance_paper_ready.png",  # name of the figure
    save_name="water_test",  # name of the saved model
    MODEL_NAME="water.eqx",  # name of the model to load
)


# ---------------------------------------- #
# For Water Modelling Test
# ---------------------------------------- #
# Hyperparams:
# hidden_size= 6
# latent_size= 2
# width_size= 16 
# depth= 2
# lr = 1e2
# theta = 0.12
# steps = 2501
# IC (1 -> 4) (1 -> 4)
# ---------------------------------------- #

