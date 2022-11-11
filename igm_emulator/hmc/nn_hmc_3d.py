from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import jax
import jax.random as random
from jax import jit
from functools import partial
from numpyro.infer import MCMC, NUTS
import arviz as az
import time
import IPython
from igm_emulator.emulator.emulator_run import nn_emulator
import sys
sys.path.insert(0,'/home/zhenyujin/dw_inference/dw_inference/inference')
from utils import walker_plot, corner_plot

class NN_HMC:
    def __init__(self, vbins, best_params, T0s, gammas, fobs, like_dict,dense_mass=True, max_tree_depth=10, num_warmup=1000, num_samples=1000, num_chains=3):
        self.vbins = vbins
        self.best_params = best_params
        self.like_dict = like_dict
        self.max_tree_depth = max_tree_depth
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.dense_mass = dense_mass
        self.mcmc_nsteps_tot = num_samples * num_chains
        self.T0s = T0s
        self.gammas = gammas
        self.fobs = fobs
        self.theta_ranges = [[self.fobs[0],self.fobs[-1]],[self.T0s[0],self.T0s[-1],[self.gammas[0],self.gammas[-1]]]


    def log_likelihood(self, theta, corr):
        ave_f, temp, g  = theta
        theta = jnp.asarray(theta)
        corr = jnp.asarray(corr)
        model = nn_emulator(self.best_params,theta)
        '''
        T0_idx_closest = np.argmin(np.abs(temps - temp))
        g_idx_closest = np.argmin(np.abs(gs - g))
        f_idx_closest = np.argmin(np.abs(average_fluxes - ave_f))
        like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx_closest}_G{g_idx_closest}_SNR0_F{f_idx_closest}_ncovar_500000_P{n_path}_set_bins_4.p'
        like_dict = dill.load(open(in_path + like_name, 'rb'))
        model_autocorrelation = like_dict['mean_data']
        '''
        new_covariance = self.like_dict['covariance']
        log_determinant = self.like_dict['log_determinant']

        diff = corr - model
        nbins = len(self.vbins)
        log_like = -(np.dot(diff, np.linalg.solve(new_covariance, diff)) + log_determinant + nbins * np.log(
            2.0 * np.pi)) / 2.0
        print(f'Log_likelihood={log_like}')
        return log_like

    def theta_to_x(self,theta):
        x_astro = []
        for theta_i, theta_range in zip(theta, self.theta_ranges):
            x_astro.append(jax.scipy.special.logit(
                jnp.clip((theta_i - theta_range[0]) / (theta_range[1] - theta_range[0]),
                         a_min=1e-7, a_max=1.0 - 1e-7)))
        return jnp.array(x_astro)

    def x_to_theta(self,x):
        theta_astro = []
        for x_i, theta_range in zip(x, self.theta_ranges):
            theta_astro.append(
                theta_astro_range[0] + (theta_range[1] - theta_range[0]) * jax.nn.sigmoid(x_i))
        return jnp.array(theta_astro)


    def log_prior(x):
        return jax.nn.log_sigmoid(x) + jnp.log(1.0 - jax.nn.sigmoid(x))

    def eval_prior(self,theta):
        print(f'prior theta:{theta}')
        prior = 0.0
        x = self.theta_to_x(theta)
        IPython.embed()
        for i in x:
            prior += self.log_prior(i)
        print(f'Prior={prior}')
        return prior

    @partial(jit, static_argnums=(0,))
    def potential_fun(self,corr,theta):
        lnPrior = self.eval_prior(theta)
        lnlike = self.log_likelihood(self, theta, corr)
        lnP = lnlike + lnPrior

        return -lnP
    IPython.embed()

    @partial(jit, static_argnums=(0,))
    def numpyro_potential_fun(self,flux):
        return partial(self.potential_fun,corr=flux)


    def mcmc_one(self, key, theta, flux):
        # Instantiate the NUTS kernel and the mcmc object
        nuts_kernel = NUTS(potential_fn=self.numpyro_potential_fun(flux),
                       adapt_step_size=True, dense_mass=True, max_tree_depth=self.max_tree_depth)
        mcmc = MCMC(nuts_kernel, num_warmup=self.num_warmup, num_samples=self.num_samples, num_chains= self.num_chains,
                chain_method='vectorized', jit_model_args=True)  # chain_method='sequential'
        # Initial position
        print(f'theta:{theta}')
        ave_f, temp, g = theta
        theta = jnp.asarray(theta)
        T0_idx_closest = np.argmin(np.abs(self.T0s - temp))
        g_idx_closest = np.argmin(np.abs(self.gammas - g))
        f_idx_closest = np.argmin(np.abs(self.fobs - ave_f))
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
        theta_samples = x_samples

        lnP = -mcmc.get_extra_fields()['potential_energy']
        hmc_num_steps = mcmc.get_extra_fields()['num_steps']  # Number of steps in the Hamiltonian trajectory (for diagnostics).
        hmc_tree_depth = np.log2(hmc_num_steps).astype(int) + 1  # Tree depth of the Hamiltonian trajectory (for diagnostics).
        hit_max_tree_depth = np.sum(
        hmc_tree_depth == 10)  # Number of transitions that hit the maximum tree depth.
        ms_per_step = 1e3 * total_time / np.sum(hmc_num_steps)

        print(f"*** SUMMARY FOR HMC ***")
        print(f"total_time = {total_time} seconds for the HMC")
        print(f"total_steps = {np.sum(hmc_num_steps)} total steps")
        print(f"ms_per_step = {ms_per_step} ms per step of the HMC")
        print(f"n_eff_mean = {neff_mean} effective sample size, compared to ntot = {self.mcmc_nsteps_tot} total samples.")
        print(f"sec_per_neff = {sec_per_neff:.3f} seconds per effective sample")
        print(f"r_hat_mean = {r_hat_mean}")
        print(f"max_tree_depth encountered = {hmc_tree_depth.max()} in chain")
        print(f"There were {hit_max_tree_depth} transitions that exceeded the max_tree_depth = {self.max_tree_depth}")
        print("*************************")

        # Return the values needed
        return x_samples, theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean, \
            hmc_num_steps, hmc_tree_depth, total_time

    def plot_HMC(self,x_samples):
        out_prefix = '/home/zhenyujin/igm_emulator/igm_emulator/hmc/plots/'
        var_label = ['fobs', 'T0s', 'gammas']
        walkerfile = out_prefix + '_walkers_' + '.pdf'
        cornerfile = out_prefix + '_corner_' + '.pdf'
        x_cornerfile = out_prefix + '_x-corner_' + '.pdf'
        specfile = out_prefix + '_spec_' + '.pdf'
        walker_plot(np.swapaxes(x_samples, 0, 1), var_label,
                    truths=self.x_true if self.x_true is not None else None,
                    walkerfile=walkerfile, linewidth=1.0)
        # Raw x_params corner plot
        corner_plot(x_samples, var_label,
                    theta_true=self.x_true if self.x_true is not None else None,
                    cornerfile=x_cornerfile)
        corner_plot(x_samples, var_label,
                    theta_true=self.theta_true if self.theta_true is not None else None,
                    cornerfile=cornerfile)
