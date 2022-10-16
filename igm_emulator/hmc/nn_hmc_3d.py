import dill
from jax.config import config
config.update("jax_enable_x64", True)
import json
import os
import numpy as np
import jax.numpy as jnp
import jax
import jax.random as random
from jax import jit
import optax
from functools import partial

from tqdm.auto import trange

import matplotlib
import matplotlib.pyplot as plt
from numpyro.infer import MCMC, NUTS
import arviz as az
import h5py
#from dw_inference.inference.utils import walker_plot, corner_plot

import time
import IPython
from igm_emulator.scripts.pytree_h5py import save, load
from igm_emulator.scripts.grab_models import param_transform
from igm_emulator.emulator.emulator_train import custom_forward
from igm_emulator.emulator.plotVis import v_bins

'''
load model and auto-corr
'''
redshift = 5.4

# get the appropriate string and pathlength for chosen redshift
zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
z_idx = np.argmin(np.abs(zs - redshift))
z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
z_string = z_strings[z_idx]
in_path_hdf5 = '/home/zhenyujin/igm_emulator/igm_emulator/emulator/best_params/'
f = h5py.File(in_path_hdf5 + f'z{redshift}_nn_savefile.hdf5', 'r')
emu_name = f'{z_string}_best_param_training_768.p'
#IPython.embed()

best_params = dill.load(open(in_path_hdf5 + emu_name, 'rb'))
meanX = np.asarray(f['data']['meanX'])
stdX = np.asarray(f['data']['stdX'])
meanY = np.asarray(f['data']['meanY'])
stdY =  np.asarray(f['data']['stdY'])
print(meanX)
#best_params = load(f)
#print(f['performance']['residuals'])
#print(f['best_params']['custom_linear/~/linear_0']['w'])
#print(load(f'/home/zhenyujin/igm_emulator/igm_emulator/emulator/best_params/z{redshift}_nn_savefile.hdf5'))

in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'
n_paths = np.array([17, 16, 16, 15, 15, 15, 14])
n_path = n_paths[z_idx]
vbins = v_bins
param_in_path = '/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/'
param_dict = dill.load(open(param_in_path + f'{z_string}_params.p', 'rb'))

fobs = param_dict['fobs']  # average observed flux <F> ~ Gamma_HI
log_T0s = param_dict['log_T0s']  # log(T_0) from temperature - density relation
T0s = np.exp(log_T0s)
gammas = param_dict['gammas']  # gamma from temperature - density relation

T0_idx = 12 #0-14
g_idx = 7 #0-8
f_idx = 7 #0-8
like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{g_idx}_SNR0_F{f_idx}_ncovar_500000_P{n_path}_set_bins_4.p'
like_dict = dill.load(open(in_path + like_name, 'rb'))
theta = [fobs[f_idx], T0s[T0_idx], gammas[g_idx]]
x_true = (theta - meanX)/ stdX
flux = like_dict['mean_data']


def log_likelihood(theta, vbins, corr, temps=T0s, gs=gammas, average_fluxes=fobs):
    ave_f, temp, g  = theta

    model = custom_forward.apply(params=best_params, x=theta)
    '''
    T0_idx_closest = np.argmin(np.abs(temps - temp))
    g_idx_closest = np.argmin(np.abs(gs - g))
    f_idx_closest = np.argmin(np.abs(average_fluxes - ave_f))
    like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx_closest}_G{g_idx_closest}_SNR0_F{f_idx_closest}_ncovar_500000_P{n_path}_set_bins_4.p'
    like_dict = dill.load(open(in_path + like_name, 'rb'))
    model_autocorrelation = like_dict['mean_data']
    '''
    new_covariance = like_dict['covariance']
    log_determinant = like_dict['log_determinant']

    diff = corr - model
    nbins = len(vbins)
    log_like = -(np.dot(diff, np.linalg.solve(new_covariance, diff)) + log_determinant + nbins * np.log(
        2.0 * np.pi)) / 2.0
    return log_like

def log_prior(x):
    return jax.nn.log_sigmoid(x) + jnp.log(1.0 - jax.nn.sigmoid(x))
def eval_prior(theta):
    prior = 0.0
    x_astro_priors= [log_prior,log_prior,log_prior]
    for x, x_astro_pri in zip(theta, x_astro_priors):
        prior += x_astro_pri(x)
    return prior

@partial(jit, static_argnums=(0,))
def potential_fun(corr,theta):
    lnPrior = eval_prior(theta)
    lnlike = log_likelihood(theta, vbins, corr)
    lnP = lnlike + lnPrior

    return -lnP

def numpyro_potential_fun(flux):
    return partial(potential_fun, flux=flux)

dense_mass=True
max_tree_depth=10
num_warmup=1000
num_samples=1000
num_chains=3
mcmc_nsteps_tot = num_samples*num_chains

def mcmc_one(key, theta, flux):
    # Instantiate the NUTS kernel and the mcmc object
    nuts_kernel = NUTS(potential_fn=numpyro_potential_fun(flux),
                       adapt_step_size=True, dense_mass=True, max_tree_depth=max_tree_depth)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains= num_chains,
                chain_method='vectorized', jit_model_args=True)  # chain_method='sequential'
    # Initial position
    ave_f, temp, g = theta
    theta = np.reshape(theta, (3,))
    T0_idx_closest = np.argmin(np.abs(T0s - temp))
    g_idx_closest = np.argmin(np.abs(gammas - g))
    f_idx_closest = np.argmin(np.abs(fobs - ave_f))
    x_opt = np.asarray([T0_idx_closest, g_idx_closest, f_idx_closest])
    # Run the MCMC
    start_time = time.time()
    IPython.embed()
    mcmc.run(key, init_params=theta, extra_fields=('potential_energy', 'num_steps'))
    total_time = time.time() - start_time

    # Compute the neff and summarize cost
    az_summary = az.summary(az.from_numpyro(mcmc))
    neff = az_summary["ess_bulk"].to_numpy()
    neff_mean = np.mean(neff)
    r_hat = az_summary["r_hat"].to_numpy()
    r_hat_mean = np.mean(r_hat)
    sec_per_neff = (total_time / neff_mean)
    # Grab the samples and lnP
    x_samples = mcmc.get_samples(group_by_chain=True) #normalized theta
    #theta_samples = param_transform(x_samples,
                                    #np.array([fobs[0], T0s[0], gammas[0]]),
                                    #np.array([fobs[-1], T0s[-1], gammas[-1]])) #real theta
    theta_samples = x_samples

    lnP = -mcmc.get_extra_fields()['potential_energy']
    hmc_num_steps = mcmc.get_extra_fields()['num_steps']  # Number of steps in the Hamiltonian trajectory (for diagnostics).
    hmc_tree_depth = np.log2(hmc_num_steps).astype(
        int) + 1  # Tree depth of the Hamiltonian trajectory (for diagnostics).
    hit_max_tree_depth = np.sum(
        hmc_tree_depth == 10)  # Number of transitions that hit the maximum tree depth.
    ms_per_step = 1e3 * total_time / np.sum(hmc_num_steps)

    print(f"*** SUMMARY FOR HMC ***")
    print(f"total_time = {total_time} seconds for the HMC")
    print(f"total_steps = {np.sum(hmc_num_steps)} total steps")
    print(f"ms_per_step = {ms_per_step} ms per step of the HMC")
    print(f"n_eff_mean = {neff_mean} effective sample size, compared to ntot = {mcmc_nsteps_tot} total samples.")
    print(f"sec_per_neff = {sec_per_neff:.3f} seconds per effective sample")
    print(f"r_hat_mean = {r_hat_mean}")
    print(f"max_tree_depth encountered = {hmc_tree_depth.max()} in chain")
    print(f"There were {hit_max_tree_depth} transitions that exceeded the max_tree_depth = {max_tree_depth}")
    print("*************************")

    # Return the values needed
    return x_samples, theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean, \
           hmc_num_steps, hmc_tree_depth, total_time



if __name__ == '__main__':
    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    x_samples, samples, ln_probs, neff, neff_mean, \
    sec_per_neff, ms_per_step, r_hat, r_hat_mean, \
    hmc_num_steps, hmc_tree_depth, runtime = mcmc_one(subkey, theta, flux)

    out_prefix = '/home/zhenyujin/igm_emulator/igm_emulator/hmc/plots/'
    walkerfile = out_prefix + '_walkers_' +  '.pdf'
    cornerfile = out_prefix + '_corner_' +  '.pdf'
    x_cornerfile = out_prefix + '_x-corner_' +  '.pdf'
    specfile = out_prefix + '_spec_' + '.pdf'
    _x_true = x_true
    _theta_true = theta
    walker_plot(np.swapaxes(x_samples, 0, 1), var_label,
                truths=self.x_true[iqso, :] if self.x_true is not None else None,
                walkerfile=walkerfile, linewidth=1.0)
    # Raw x_params corner plot
    corner_plot(x_samples, x_var_label,
                theta_true=_x_true if self.x_true is not None else None,
                cornerfile=x_cornerfile)
    corner_plot(samples, var_label,
                theta_true=self.theta_true if self.theta_true is not None else None,
                cornerfile=cornerfile)