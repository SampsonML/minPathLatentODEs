# ---------------------------------------- #
# MLP for parameter inference on the LVE   #
# test. Use the context vector and latents #
# generated by the LatentODE and infer the #
# model parameters.                        #
# ---------------------------------------- #

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
import corner

# debugging
print(f"Optax version {optax.__version__}")

# import LVE model things
from latentPathMin import LatentODE, get_data, dataloader

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


# --------------------
# Define the MLP model
class inferenceMLP(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, key, **kwargs):
        super().__init__(**kwargs)
        # Define the MLP
        self.mlp = eqx.nn.MLP(
            in_size=input_dim,
            out_size=output_dim,
            width_size=hidden_dim,
            depth=num_layers,
            activation=jnn.softplus,
            key=key,
        )

    def __call__(self, x):
        return self.mlp(x)

    # define a loss function inside model
    # for now just predict with initial latent variables
    @staticmethod
    def _loss(params, preds):
        loss = jnp.mean((preds - params) ** 2)
        return loss 

    # define a train call for the model 
    def train(self, params, context, latents):
        preds = self.mlp(latents)
        return self._loss(params, preds)


# ------------------------
# Define the training step

# get pseudo RNG keys
key = jr.PRNGKey(1992)
data_key, model_key, loader_key, train_key, sample_key = jr.split(key, 5)

# get the data
bounds = [(1.0, 2.0), (1.0, 2.0), (0.5, 1.0), (0.5, 1.0)]  # LVE param bounds
IC_min = 2
IC_max = 4 
n_points=100 
dataset_size = 5000
t_final = 20
ts, ys, params, ICs = get_data(
    dataset_size,
    key=data_key,
    bounds=bounds,
    t_end=t_final,
    n_points=n_points,
    IC_min=IC_min,
    IC_max=IC_max,
)


# make a test split -- randomly select 10% of the data for testing
test_size = int(0.1 * dataset_size)
test_idx = jr.choice(data_key, dataset_size, (test_size,), replace=False)
train_idx = jnp.setdiff1d(jnp.arange(dataset_size), test_idx)
ts_train, ys_train, params_train, ICs_train = (
    ts[train_idx],
    ys[train_idx],
    params[train_idx],
    ICs[train_idx],
)
ts_test, ys_test, params_test, ICs_test = (
    ts[test_idx],
    ys[test_idx],
    params[test_idx],
    ICs[test_idx],
)

# load the trained latentODE-RNN model
MODEL_NAME = "lve_new_a1__npoints_250_hsz4_lsz4_w60_d3_lossTypemahalanobis_step_12000.eqx"
ODEhidden_size = 4
ODElatent_size = 4
ODEwidth_size = 60
ODEdepth = 3 
alpha = 2
key = jr.PRNGKey(1992)


ODEmod = LatentODE(
    data_size=ys.shape[-1],
    hidden_size=ODEhidden_size,
    latent_size=ODElatent_size,
    width_size=ODEwidth_size,
    depth=ODEdepth,
    key=model_key,
    alpha=alpha,
    lossType="mahalanobis",
)

ODEmodel = eqx.tree_deserialise_leaves("trainedLatentODEs/" + MODEL_NAME, ODEmod)
latent, _, _, _, = ODEmodel._latent(ts_train[0,:], ys_train[0,:,:], model_key)

# ------------------------
# Instantiate the MLP model
num_params = 4  # for now hard coding the number of parameters in LVE model exclude ICs
hidden_dim = 32
input_size = latent.shape[-1]  # size of the latent vector from the LatentODE model
num_layers = 3
model = inferenceMLP(
    input_dim=input_size,
    output_dim=num_params,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    key=model_key,
)

# create the loss function
@eqx.filter_value_and_grad
def loss(params, model, context, latent):
    batch_size, _ = ts_i.shape
    loss = jax.vmap(model.train)(params, context, latent)
    return jnp.mean(loss)


#@eqx.filter_jit
def make_step(model, opt_state, params, context, latents):
    value, grads = loss(params, model, context, latents)
    preds = jax.vmap(model)(latents)
    #print(f"Preds: {preds}")
    #print(f"Value: {value}")
    print(f"Grads: {grads}")
    #print(f"Opt state: {opt_state}")
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return value, model, opt_state


# training hyperparams
batch_size=5
steps = 100
lr=1e-2
train = True
loss_vector = []

# model save/store options
SAVE_DIR = "saved_models"
save_name = "inferenceMLP"
save_every = steps // 10

# initialize the optimizer
optim = optax.adam(lr)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

# run training loop
for step, (ts_i, ys_i, params_i, ICs_i) in zip(
    range(steps),
    dataloader(
        (ts_train, ys_train, params_train, ICs_train), batch_size, key=loader_key
    ),
):
    if train:
        start = time.time()
        # get the context and latents from the trained LatentODE model
        ODE_key = jr.split(model_key, ts_i.shape[0])
        latent, _, _, context = jax.vmap(ODEmodel._latent)(ts_i, ys_i, ODE_key)
        value, model, opt_state, = make_step(
            model,
            opt_state,
            params_i,
            context,
            latent,
        )
        end = time.time()
        print(f"Step: {step}, Loss: {value}, Computation time: {end - start}")
        loss_vector.append(value)

        # save the model
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        if (step % save_every) == 0 or step == steps - 1:
            fn = SAVE_DIR + "/" + save_name + "_step_" + str(step) + ".eqx"
            eqx.tree_serialise_leaves(fn, model)

    # load the model instead here
    else:
        modelName = "saved_models/" + MODEL_NAME
        model = eqx.tree_deserialise_leaves(modelName, model)


# ------------------------------------------------------------------------------- #
# ------------------------ PLOTTING AND INFERENCE ------------------------------- #
# ------------------------------------------------------------------------------- #
# plot the loss 
fig = plt.figure(figsize=(8, 6))
plt.plot(loss_vector, color='darkorchid', lw=2)
plt.xlabel("Step", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.savefig("inferenceMLP_loss.png", dpi=300)

# test the inference capabilities for a single fixed param/IC combo
bounds = [(1.0, 1.0), (1.0, 1.0), (0.5, 5.0), (0.5, 0.5)]  # LVE param bounds
IC_min = 3
IC_max = 3
n_trials = 50
t_final = 20
ts, ys, params, ICs = get_data(
    n_trials,
    key=data_key,
    bounds=bounds,
    t_end=t_final,
    n_points=n_points,
    IC_min=IC_min,
    IC_max=IC_max,
)

# get the context and latents from the trained LatentODE model
ODE_key = jr.split(model_key, n_trials)
latent_test, _, _, context_test = jax.vmap(ODEmodel)(ts_test, ys_test, ODE_key)

# get the parameter predictions
params_pred = jax.vmap(model)(latent_test)
labels = [r'$\alpha$', r'$\Beta$', r'$\gamma$', r'$\delta$']
fig = corner.corner(np.array(params_pred), labels=labels, truths=np.array(params_test))
plt.savefig("inferenceMLP_corner.png", dpi=300)


