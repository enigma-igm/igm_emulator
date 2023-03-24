from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import jax
import h5py
import jax.random as random
from jax import jit
from functools import partial
from numpyro.infer import MCMC, NUTS
import arviz as az
import time
import IPython
import os
import sys
from igm_emulator.emulator.emulator_run import nn_emulator
sys.path.append(os.path.expanduser('~') + '/dw_inference/dw_inference/inference')
from utils import walker_plot, corner_plot
import struct
print(struct.calcsize("P") * 8)

#running everything in dimensionless parameter space (x)
class NN_HMC_X:
    def __init__(self, vbins, best_params, T0s, gammas, fobs, like_dict,dense_mass=True, max_tree_depth=(8,10), num_warmup=1000, num_samples=1000, num_chains=4):
        '''
        Args:
            vbins: velocity bins
            best_params: best parameters from the neural network
            T0s: temperature array
            gammas: gamma array
            fobs: frequency array
            like_dict: dictionary containing the covariance matrix, log determinant, and inverse covariance matrix
            dense_mass: whether to use dense mass matrix
            max_tree_depth: maximum tree depth
            num_warmup: number of warmup steps
        num_samples: number of samples
                num_chains: number of chains

        Returns:
            samples: samples from the posterior
        '''
        self.vbins = vbins
        self.best_params = best_params
        self.like_dict = like_dict
        self.max_tree_depth = max_tree_depth
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.dense_mass = dense_mass
        self.mcmc_nsteps_tot = num_samples * num_chains
        self.num_samples = num_samples
        self.T0s = T0s
        self.gammas = gammas
        self.fobs = fobs
        self.theta_ranges = [[self.fobs[0],self.fobs[-1]],[self.T0s[0],self.T0s[-1]],[self.gammas[0],self.gammas[-1]]]

    @partial(jit, static_argnums=(0,))
    def log_likelihood(self, x, corr):
        '''
        Args:
            x: dimensionless parameters
            flux: observed flux

        Returns:
            log_likelihood: log likelihood
        '''
        theta = self.x_to_theta(x)
        model = nn_emulator(self.best_params,theta) #theta is in physical dimension for this function

        new_covariance = self.like_dict['covariance']
        log_determinant = self.like_dict['log_determinant']

        diff = corr - model
        nbins = len(self.vbins)
        log_like = -(jnp.dot(diff, jnp.linalg.solve(new_covariance, diff)) + log_determinant + nbins * jnp.log(
            2.0 * jnp.pi)) / 2.0
        #print(f'Log_likelihood={log_like}')
        return log_like

    @partial(jit, static_argnums=(0,))
    def _theta_to_x(self,theta): #theta is in physical dimension
        x_astro = []
        for theta_i, theta_range in zip(theta, self.theta_ranges):
            x_astro.append(jax.scipy.special.logit(
                jnp.clip((theta_i - theta_range[0]) / (theta_range[1] - theta_range[0]),
                         a_min=1e-7, a_max=1.0 - 1e-7)))
        return jnp.array(x_astro)

    @partial(jit, static_argnums=(0,))
    def theta_to_x(self, theta, axis=0): #x is in dimensionless parameter space

        x_astro = jax.vmap(
            self._theta_to_x, in_axes=axis, out_axes=axis)(jnp.atleast_2d(theta))

        return x_astro.squeeze()

    @partial(jit, static_argnums=(0,))
    def _x_to_theta(self,x):
        theta_astro = []
        for x_i, theta_range in zip(x, self.theta_ranges):
            theta_astro.append(theta_range[0] + (theta_range[1] - theta_range[0]) * jax.nn.sigmoid(x_i))
        return jnp.array(theta_astro)

    @partial(jit, static_argnums=(0,))
    def x_to_theta(self, x, axis=0):

        theta_astro = jax.vmap(self._x_to_theta, in_axes=axis, out_axes=axis)(jnp.atleast_2d(x))

        return theta_astro.squeeze()

    def log_prior(self,x):
        '''
        Args:
            x: dimensionless parameters

        Returns:
            log_prior: log prior
        '''
        return jax.nn.log_sigmoid(x) + jnp.log(1.0 - jax.nn.sigmoid(x))

    @partial(jit, static_argnums=(0,))
    def eval_prior(self,x):
        prior = 0.0
        for i in x:
            prior += self.log_prior(i)
            #print(f'i={i}')
        #print(f'Prior={prior}')
        return prior

    @partial(jit, static_argnums=(0,))
    def potential_fun(self,x,flux):
        '''
        Parameters
        ----------
        self
        x: dimensionless parameters
        flux: observed flux

        Returns
        -------
        lnP: log posterior
        '''
        #in physical space
        lnPrior = self.eval_prior(x)
        lnlike = self.log_likelihood(x, flux)
        lnP = lnlike + lnPrior

        return -lnP

    @partial(jit, static_argnums=(0,))
    def numpyro_potential_fun(self, flux): #potential function for numpyro
        return jax.tree_util.Partial(self.potential_fun, flux=flux)


    def mcmc_one(self, key, x, flux): #input dimensionless paramter x
        '''

        Parameters
        ----------
        self
        key: random key
        x: dimensionless parameters
        flux: observed flux

        Returns
        -------
        x_samples: samples from the posterior
        theta_samples: samples from the posterior in physical space
        lnP: log posterior
        neff: effective sample size
        neff_mean: mean effective sample size
        sec_per_neff: time per effective sample
        ms_per_step: time per step
        r_hat: Gelman-Rubin statistic
        r_hat_mean : mean Gelman-Rubin statistic
        hmc_num_steps: number of steps in the HMC
        hmc_tree_depth: depth of the tree in the HMC
        total_time: total time
        '''
        # Instantiate the NUTS kernel and the mcmc object
        nuts_kernel = NUTS(potential_fn=self.numpyro_potential_fun(flux),
                       adapt_step_size=True, dense_mass=True, max_tree_depth=self.max_tree_depth)
        mcmc = MCMC(nuts_kernel, num_warmup=self.num_warmup, num_samples=self.num_samples, num_chains= self.num_chains,
                 jit_model_args=True, chain_method='vectorized')  # chain_method='sequential' chain_method='vectorized'
        # Initial position
        print(f'true theta:{self.x_to_theta(x)}')
        print(f'true x:{x}')
        theta = self.x_to_theta(x)
        #ave_f, temp, g = theta
        #T0_idx_closest = np.argmin(np.abs(self.T0s - temp))
        #g_idx_closest = np.argmin(np.abs(self.gammas - g))
        #f_idx_closest = np.argmin(np.abs(self.fobs - ave_f))
        theta_init = theta + 1e-4 * np.random.randn(self.num_chains, 3)
        x_init = x + 1e-4 * np.random.randn(self.num_chains, 3)
        #print(f'init pos:{theta_init}')
        
        # Run the MCMC
        start_time = time.time()
        #IPython.embed()
        mcmc.run(key, init_params=x_init.squeeze(), extra_fields=('potential_energy', 'num_steps'))
        total_time = time.time() - start_time

        # Compute the neff and summarize cost
        az_summary = az.summary(az.from_numpyro(mcmc))
        neff = az_summary["ess_bulk"].to_numpy()
        neff_mean = np.mean(neff)
        r_hat = az_summary["r_hat"].to_numpy()
        r_hat_mean = np.mean(r_hat)
        sec_per_neff = (total_time / neff_mean)
        ms_per_neff = 1e3 * sec_per_neff
        
        # Grab the samples and lnP
        theta_samples = self.x_to_theta(mcmc.get_samples(group_by_chain=False)) #(mcmc_nsteps_tot, ndim)
        x_samples = mcmc.get_samples(group_by_chain=True)#(num_chain, num_samples, ndim)
        lnP = -mcmc.get_extra_fields()['potential_energy']
        hmc_num_steps = mcmc.get_extra_fields()['num_steps']  # Number of steps in the Hamiltonian trajectory (for diagnostics).
        hmc_tree_depth = np.log2(hmc_num_steps).astype(int) + 1  # Tree depth of the Hamiltonian trajectory (for diagnostics).
        hit_max_tree_depth = np.sum(
        hmc_tree_depth == self.max_tree_depth)  # Number of transitions that hit the maximum tree depth.
        ms_per_step = 1e3 * total_time / np.sum(hmc_num_steps)

        print(f"*** SUMMARY FOR HMC ***")
        print(f"total_time = {total_time} seconds for the HMC")
        print(f"total_steps = {np.sum(hmc_num_steps)} total steps")
        print(f"ms_per_step = {ms_per_step} ms per step of the HMC")
        print(f"n_eff_mean = {neff_mean} effective sample size, compared to ntot = {self.mcmc_nsteps_tot} total samples.")
        print(f"ms_per_neff = {ms_per_neff:.3f} ms per effective sample")
        print(f"r_hat_mean = {r_hat_mean}")
        print(f"max_tree_depth encountered = {hmc_tree_depth.max()} in chain")
        print(f"There were {hit_max_tree_depth} transitions that exceeded the max_tree_depth = {self.max_tree_depth}")
        print("*************************")

        # Return the values needed
        return x_samples, theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean, \
            hmc_num_steps, hmc_tree_depth, total_time

    def plot_HMC(self,x_samples,theta_samples,theta,note):
        var_label = ['fobs', 'T0s', 'gammas']
        walkerfile = out_prefix + '_walkers_' + note + '.pdf'
        cornerfile = out_prefix + '_corner_' + note + '.pdf'
        x_cornerfile = out_prefix + '_x-corner_' + '.pdf'
        specfile = out_prefix + '_spec_' + '.pdf'
        walker_plot(np.swapaxes(x_samples, 0, 1), var_label,
                    truths= self.theta_to_x(theta),
                    walkerfile=walkerfile, linewidth=1.0) #in dimensiless space
        corner_plot(theta_samples, var_label,
                    theta_true=jnp.asarray(theta),
                    cornerfile=cornerfile)

    def save_HMC(self,zstr,mock_idx,f_idx,T0_idx,g_idx, f_mcmc, t_mcmc, g_mcmc, x_samples, theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean,
                 hmc_num_steps, hmc_tree_depth, total_time):
        # Save the results
        with h5py.File('/mnt/quasar2/zhenyujin/igm_emulator/hmc' + f'{zstr}_F{f_idx}_T0{T0_idx}_G{g_idx}_hmc_results_mock{mock_idx}.hdf5', 'w') as f:
            f.create_dataset('x_samples', data=x_samples)
            f.create_dataset('theta_samples', data=theta_samples)
            f.create_dataset('lnP', data=lnP)
            f.create_dataset('neff', data=neff)
            f.create_dataset('neff_mean', data=neff_mean)
            f.create_dataset('sec_per_neff', data=sec_per_neff)
            f.create_dataset('ms_per_step', data=ms_per_step)
            f.create_dataset('r_hat', data=r_hat)
            f.create_dataset('r_hat_mean', data=r_hat_mean)
            f.create_dataset('hmc_num_steps', data=hmc_num_steps)
            f.create_dataset('hmc_tree_depth', data=hmc_tree_depth)
            f.create_dataset('total_time', data=total_time)
            f.create_dataset('f_infer', data=f_mcmc)
            f.create_dataset('t_infer', data=t_mcmc)
            f.create_dataset('g_infer', data=g_mcmc)
            f.close()
        print(f"hmc results saved for mock {mock_idx} corr")