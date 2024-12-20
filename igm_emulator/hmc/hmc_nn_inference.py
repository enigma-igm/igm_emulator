import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import h5py
import jax.random as random
from jax import jit
from jax.scipy.stats.multivariate_normal import logpdf
import optax
from tqdm.auto import trange
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_percentage_error
from scipy.spatial.distance import minkowski
from functools import partial
from numpyro.infer import MCMC, NUTS
import arviz as az
import time
import IPython
import igm_emulator as emu
import sys
import os
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator')
from emulator_apply import nn_emulator
sys.path.append(os.path.expanduser('~') + '/qso_fitting/')
from qso_fitting.fitting.utils import bounded_theta_to_x, x_to_bounded_theta, bounded_variable_lnP
import corner
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as pe
from tabulate import tabulate
import struct

x_size = 3.5
dpi_value = 200

plt_params = {'legend.fontsize': 7,
              'legend.frameon': False,
              'axes.labelsize': 8,
              'axes.titlesize': 8,
              'figure.titlesize': 7,
              'xtick.labelsize': 7,
              'ytick.labelsize': 7,
              'lines.linewidth': .7,
              'lines.markersize': 2.3,
              'lines.markeredgewidth': .9,
              'errorbar.capsize': 2,
              'font.family': 'serif',
              # 'text.usetex': True,
              'xtick.minor.visible': True,
              }
plt.rcParams.update(plt_params)

#running everything in dimensionless parameter space (x)
class NN_HMC_X:
    def __init__(self, vbins, best_params, T0s, gammas, fobs, dense_mass=True,
                 max_tree_depth= 10, #(8,10),
                 num_warmup= 1000,
                 num_samples= 1000,
                 num_chains= 4,
                 opt_nsteps=150,
                 opt_lr=0.01,
                 covar_nn = None,
                 err_nn = None,
                 nn_err_prop = False
                 ):
        '''
        Args:
            vbins: velocity bins
            best_params: best parameters from the neural network
            T0s: temperature array
            gammas: gamma array
            fobs: frequency array
            dense_mass: whether to use dense mass matrix
            max_tree_depth: maximum tree depth
            num_warmup: number of warmup steps
        num_samples: number of samples
                num_chains: number of chains

        Returns:
            samples: samples from the posterior
        '''
        #Set the neural network parameters
        self.vbins = vbins
        self.best_params = best_params

        #Set the HMC parameters
        self.max_tree_depth = max_tree_depth
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.dense_mass = dense_mass
        self.mcmc_nsteps_tot = num_samples * num_chains
        self.num_samples = num_samples

        #Set the prior ranges
        self.T0s = T0s
        self.gammas = gammas
        self.fobs = fobs
        self.theta_ranges = [[self.fobs[0],self.fobs[-1]],[self.T0s[0],self.T0s[-1]],[self.gammas[0],self.gammas[-1]]]
        self.theta_astro_inits = tuple([np.mean([tup[0], tup[1]]) for tup in self.theta_ranges])
        self.theta_mins = jnp.array([astro_par_range[0] for astro_par_range in self.theta_ranges])
        self.theta_maxs = jnp.array([astro_par_range[1] for astro_par_range in self.theta_ranges])
        self.x_astro_priors = [bounded_variable_lnP, bounded_variable_lnP, bounded_variable_lnP]

        # Set the optimizer parameters
        self.opt_nsteps = opt_nsteps
        self.opt_lr = opt_lr

        #Set the covariance matrix from NN-error propoganation
        self.covar_nn = covar_nn
        self.err_nn = err_nn
        self.nn_err_prop = nn_err_prop

    @partial(jit, static_argnums=(0,))
    def get_model_nearest_fine(
            self, theta
    ):
        model = emu.nn_emulator(self.best_params, theta)

        return model

    @partial(jit, static_argnums=(0,))
    def log_likelihood(self, x, corr, covar):
        '''
        Args:
            x: dimensionless parameters
            corr: observed flux
            covar: model-dependent covariance matrix

        Returns:
            log_likelihood: log likelihood to maximize on
        '''
        theta = self.x_to_theta(x)
        model = self.get_model_nearest_fine(theta) #theta is in physical dimension for this function


        log_like = logpdf(x=corr, mean=model, cov=covar)
        #print(f'Log_likelihood={log_like}')
        return log_like

    @partial(jit, static_argnums=(0,))
    def log_likelihood_nn_error(self, x, corr, covar):
        '''
        Args:
            x: dimensionless parameters
            corr: observed flux
            covar: model-dependent covariance matrix

        Returns:
            log_likelihood: log likelihood to maximize on
        '''
        theta = self.x_to_theta(x)
        model = self.get_model_nearest_fine(theta) #theta is in physical dimension for this function
        covar += self.covar_nn

        log_like = logpdf(x=corr-self.err_nn, mean=model, cov=covar)
        return log_like

    @partial(jit, static_argnums=(0,))
    def _theta_to_x(self,theta): #theta is in physical dimension
        x_astro = bounded_theta_to_x(theta, self.theta_ranges)
        return jnp.array(x_astro)

    @partial(jit, static_argnums=(0,))
    def theta_to_x(self, theta, axis=0): #x is in dimensionless parameter space
        '''
        Transform theta (nsamples, n_params) or (n_params,) to x: [fob, T0, gamma]
        '''
        x_astro = jax.vmap(self._theta_to_x, in_axes=axis, out_axes=axis)(jnp.atleast_2d(theta))

        return x_astro.squeeze()

    @partial(jit, static_argnums=(0,))
    def _x_to_theta(self,x):
        theta_astro = x_to_bounded_theta(x, self.theta_ranges)
        return jnp.array(theta_astro)

    @partial(jit, static_argnums=(0,))
    def x_to_theta(self, x, axis=0):
        '''
        Transform x (nsamples, n_params) or (n_params,) to theta: [fob, T0, gamma]
        '''
        theta_astro = jax.vmap(self._x_to_theta, in_axes=axis, out_axes=axis)(jnp.atleast_2d(x))

        return theta_astro.squeeze()

    @partial(jit, static_argnums=(0,))
    def eval_prior(self, x_astro):
        """
        Compute the prior on astro_params

        Args:
            x_astro (ndarray): shape = (nastro,)
                dimensionless astrophysical parameter vector

        Returns:
            prior (float):
                Prior on these model parameters
        """

        prior = 0.0
        for x_ast, x_astro_pri in zip(x_astro, self.x_astro_priors):
            prior += x_astro_pri(x_ast)

        return prior

    @partial(jit, static_argnums=(0,))
    def potential_fun(self,x,flux,covar):
        '''
        Parameters
        ----------
        self
        x: dimensionless parameters
        flux: observed flux

        Returns
        -------
        lnP: log posterior to maximize on
        '''
        #in physical space
        lnPrior = self.eval_prior(x)
        lnlike = self.log_likelihood(x, flux, covar)
        if self.nn_err_prop:
            lnlike = self.log_likelihood_nn_error(x, flux, covar)
        lnP = lnlike + lnPrior
        return -lnP

    @partial(jit, static_argnums=(0,))
    def numpyro_potential_fun(self, flux, covar): #potential function for numpyro
        return jax.tree_util.Partial(self.potential_fun, flux=flux, covar=covar)

    def explore_logP_plot(self, z_string, theta_true, flux, covar, fix='t', save_str=None, plot=['logP', 'logPrior', 'lnlike', 'chi']):
        """
        Explore the negative of the Potential function (prop to logL + logPrior by plotting it as a
        function of the parameters.
        Args:
            theta_plot (ndarray):
                true theta; shape (3, ) where we plot the logP around
            flux (ndarray):
                autocorrelation function data; shape (n_data, )
        """
        # create a grid of for the theta parameters
        f_grid = np.linspace(self.theta_ranges[0][0], self.theta_ranges[0][1], 100)
        t_grid = np.linspace(self.theta_ranges[1][0], self.theta_ranges[1][1], 100)
        g_grid = np.linspace(self.theta_ranges[2][0], self.theta_ranges[2][1], 100)
        closest_temp_idx = np.argmin(np.abs(self.T0s - theta_true[1]))
        closest_gamma_idx = np.argmin(np.abs(self.gammas - theta_true[2]))
        closest_fobs_idx = np.argmin(np.abs(self.fobs - theta_true[0]))

        if fix == 't':
            x_grid=f_grid
            x_label = 'f_grid'
            y_grid=g_grid
            y_label = 'g_grid'
            # create the empty array with the likelihood values
            logP_grid = np.zeros((len(x_grid), len(y_grid)))
            lnPrior_grid = np.zeros((len(x_grid), len(y_grid)))
            lnlike_grid = np.zeros((len(x_grid), len(y_grid)))
            chi_grid = np.zeros((len(x_grid), len(y_grid)))
            # loop over the grid and compute the likelihood
            t = theta_true[1]
            for i, f in enumerate(x_grid):
                for j, g in enumerate(y_grid):
                    logP_grid[i, j] = -self.potential_fun(self.theta_to_x(np.array([f, t, g])),
                                                          # change the order of f,t,g
                                                          flux, covar)
                    lnPrior_grid[i, j] = self.eval_prior(self.theta_to_x(np.array([f, t, g])))
                    lnlike_grid[i, j] = self.log_likelihood(self.theta_to_x(np.array([f, t, g])), flux, covar)
                    chi_grid[i, j] = jnp.mean(jnp.sqrt((self.get_model_nearest_fine(np.array([f, t, g]))-flux)**2)/ jnp.sqrt(jnp.diagonal(covar)))

        elif fix == 'f':
            x_grid=t_grid
            x_label = 't_grid'
            y_grid=g_grid
            y_label = 'g_grid'
            # create the empty array with the likelihood values
            logP_grid = np.zeros((len(x_grid), len(y_grid)))
            lnPrior_grid = np.zeros((len(x_grid), len(y_grid)))
            lnlike_grid = np.zeros((len(x_grid), len(y_grid)))
            chi_grid = np.zeros((len(x_grid), len(y_grid)))
            # loop over the grid and compute the likelihood
            f = theta_true[0]
            for i, t in enumerate(x_grid):
                for j, g in enumerate(y_grid):
                    logP_grid[i, j] = -self.potential_fun(self.theta_to_x(np.array([f, t, g])),
                                                          # change the order of f,t,g
                                                          flux, covar)
                    lnPrior_grid[i, j] = self.eval_prior(self.theta_to_x(np.array([f, t, g])))
                    lnlike_grid[i, j] = self.log_likelihood(self.theta_to_x(np.array([f, t, g])), flux, covar)
                    chi_grid[i, j] = jnp.mean(jnp.sqrt((self.get_model_nearest_fine(np.array([f, t, g]))-flux)**2)/ jnp.sqrt(jnp.diagonal(covar)))

        elif fix == 'g':
            x_grid=f_grid
            x_label = 'f_grid'
            y_grid=t_grid
            y_label = 't_grid'
            # create the empty array with the likelihood values
            logP_grid = np.zeros((len(x_grid), len(y_grid)))
            lnPrior_grid = np.zeros((len(x_grid), len(y_grid)))
            lnlike_grid = np.zeros((len(x_grid), len(y_grid)))
            chi_grid = np.zeros((len(x_grid), len(y_grid)))
            # loop over the grid and compute the likelihood
            g = theta_true[2]
            for i, f in enumerate(x_grid):
                for j, t in enumerate(y_grid):
                    logP_grid[i, j] = -self.potential_fun(self.theta_to_x(np.array([f, t, g])),
                                                          # change the order of f,t,g
                                                          flux, covar)
                    lnPrior_grid[i, j] = self.eval_prior(self.theta_to_x(np.array([f, t, g])))
                    lnlike_grid[i, j] = self.log_likelihood(self.theta_to_x(np.array([f, t, g])), flux, covar)
                    chi_grid[i, j] = jnp.mean(jnp.sqrt((self.get_model_nearest_fine(np.array([f, t, g]))-flux)**2)/ jnp.sqrt(jnp.diagonal(covar)))

        # plot the log_posterior
        if 'logP' in plot:
            plt.figure(figsize=(10, 8))
            plt.imshow(logP_grid, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], origin='lower', aspect='auto')
            plt.colorbar(label='logP')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title('Color plot of logP_grid')
            plt.savefig(f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{z_string}/hmc/logP_grid_fix_{fix}_T{closest_temp_idx}_G{closest_gamma_idx}_F{closest_fobs_idx}_{save_str}.pdf')
            plt.close()
        #plot the log_prior
        if 'logPrior' in plot:
            plt.figure(figsize=(10, 8))
            plt.imshow(lnPrior_grid, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], origin='lower', aspect='auto')
            plt.colorbar(label='lnPrior')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title('Color plot of lnPrior_grid')
            plt.savefig(f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{z_string}/hmc/lnPrior_grid_fix_{fix}_T{closest_temp_idx}_G{closest_gamma_idx}_F{closest_fobs_idx}_{save_str}.pdf')
            plt.close()
        #plot the log_likelihood
        if 'lnlike' in plot:
            plt.figure(figsize=(10, 8))
            plt.imshow(lnlike_grid, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], origin='lower', aspect='auto')
            plt.colorbar(label='lnlike')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title('Color plot of lnlike_grid')
            plt.savefig(f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{z_string}/hmc/lnlike_grid_fix_{fix}_T{closest_temp_idx}_G{closest_gamma_idx}_F{closest_fobs_idx}_{save_str}.pdf')
            plt.close()
        #plot the difference squared
        if 'chi' in plot:
            plt.figure(figsize=(10, 8))
            plt.imshow(chi_grid, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], origin='lower', aspect='auto', cmap='viridis_r')
            plt.colorbar(label='chi',spacing='proportional',format = '%.4e')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f'Color plot of mean chi; min chi:{chi_grid.min()}')
            plt.savefig(f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{z_string}/hmc/chi_grid_fix_{fix}_T{closest_temp_idx}_G{closest_gamma_idx}_F{closest_fobs_idx}_{save_str}.pdf')
            plt.close()
        return fix, f_grid, t_grid, g_grid, logP_grid, chi_grid

#### functions to do the MCMC initialization
    def x_minmax(self):
        x_min, x_max = bounded_theta_to_x(self.theta_mins, self.theta_ranges), \
            bounded_theta_to_x(self.theta_maxs, self.theta_ranges)

        return x_min, x_max
    def mcmc_init_x(self,key, perturb, x_opt):
        """

        Args:
            perturb:
            x_opt:

        Returns:

        """

        x_min, x_max = self.x_minmax()
        delta_x = x_max - x_min

        key, subkey = random.split(key)
        deviates = perturb * random.normal(subkey, (self.num_chains, 3))

        x_init = jnp.array([[jnp.clip(x_opt[i] + delta_x[i] * deviates[j, i],
                                          x_min[i], x_max[i]) for i in range(3)]
                                for j in range(self.num_chains)])

        return x_init.squeeze()

    def fit_one(self, flux, ivar):
        x = self.theta_to_x(self.theta_astro_inits)
        optimizer = optax.adam(self.opt_lr)
        opt_state = optimizer.init(x)
        losses = np.zeros(self.opt_nsteps)
        # Optimization loop for fitting input flux
        iterator = trange(self.opt_nsteps, leave=False)
        best_loss = np.inf  # Models are only saved if they reduce the validation loss
        for i in iterator:
            losses[i], grads = jax.value_and_grad(self.potential_fun, argnums=0)(x, flux, ivar)
            if losses[i] < best_loss:
                x_out = x.copy()
                theta_out = self.x_to_theta(x_out)
                best_loss = losses[i]
            updates, opt_state = optimizer.update(grads, opt_state)
            x = optax.apply_updates(x, updates)

        return x_out, theta_out, losses
    ####
    def mcmc_one(self, key, x, flux, covar, report): #input dimensionless paramter x
        '''

        Parameters
        ----------
        self
        key: random key, need to random.split(key) everytime before mcmc_one
        x: dimensionless best-fit parameters
        flux: observed flux
        covar: model-dependent covariance matrix
        report: whether to report the progress

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
        nuts_kernel = NUTS(potential_fn=self.numpyro_potential_fun(flux,covar),
                       adapt_step_size=True, dense_mass=True, max_tree_depth=self.max_tree_depth,
                        find_heuristic_step_size=True, target_accept_prob=0.9)
        # Initial position
        if report:
            print(f'opt theta:{self.x_to_theta(x)}')
            print(f'opt x:{x}')
            mcmc = MCMC(nuts_kernel, num_warmup=self.num_warmup, num_samples=self.num_samples,
                        num_chains=self.num_chains,
                        jit_model_args=True,
                        chain_method='vectorized')  # chain_method='sequential' chain_method='vectorized' regularize_mass_matrix=False
        else:
            mcmc = MCMC(nuts_kernel, num_warmup=self.num_warmup, num_samples=self.num_samples,
                        num_chains=self.num_chains,
                        jit_model_args=True,
                        chain_method='vectorized',
                        progress_bar=False)  # chain_method='sequential' chain_method='vectorized'
        theta = self.x_to_theta(x)
        theta_init = theta + 0.05 * np.random.randn(self.num_chains, 3)
        #x_init = x + 1e-4 * np.random.randn(self.num_chains, 3)
        x_init = self.mcmc_init_x(key,  0.05, x)
        
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
        hit_max_tree_depth = np.sum(hmc_tree_depth == self.max_tree_depth)  # Number of transitions that hit the maximum tree depth.
        ms_per_step = 1e3 * total_time / np.sum(hmc_num_steps)

        if report:
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

    def save_HMC(self,zstr,f_idx,T0_idx,g_idx, f_mcmc, t_mcmc, g_mcmc, x_samples, theta_samples, theta_true, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean,
                 hmc_num_steps, hmc_tree_depth, total_time,save_str=None, ):
        if os.path.exists(os.path.expanduser('~') + f'/igm_emulator/igm_emulator/hmc/hmc_results/{zstr}_F{f_idx}_T0{T0_idx}_G{g_idx}_{save_str}_hmc_results.hdf5'):
            os.remove(os.path.expanduser('~') + f'/igm_emulator/igm_emulator/hmc/hmc_results/{zstr}_F{f_idx}_T0{T0_idx}_G{g_idx}_{save_str}_hmc_results.hdf5')

        # Save the results
        with h5py.File(os.path.expanduser('~') + f'/igm_emulator/igm_emulator/hmc/hmc_results/{zstr}_F{f_idx}_T0{T0_idx}_G{g_idx}_{save_str}_hmc_results.hdf5', 'w') as f:
            f.create_dataset('theta_true', data=theta_true)
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
        print(f"hmc results saved for {zstr}_F{f_idx}_T0{T0_idx}_G{g_idx}_{save_str}")

    def corner_plot(self,z_string,theta_samples,x_samples,theta_true,save_str=None, save_bool=False):
        '''
        Plot the corner plot for the HMC results
        Parameters
        ----------
        z_string: str of redshift ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
        theta_samples: list of samples from HMC in theta space, shape (nwalkers * nsteps, ndim)
        x_samples: list of samples from HMC in x space, shape (nwalkers, nsteps, ndim)
        theta_true: list of true values of theta [fob, T0, gamma], shape (ndim,)

        Returns
        -------
        corner_fig_theta: corner plot for theta
        corner_fig_x: corner plot for x
        '''

        closest_temp_idx = np.argmin(np.abs(self.T0s - theta_true[1]))
        closest_gamma_idx = np.argmin(np.abs(self.gammas - theta_true[2]))
        closest_fobs_idx = np.argmin(np.abs(self.fobs - theta_true[0]))
        var_label = [r'$\langle F \rangle$', r'$T_0$', r'$\gamma$']

        corner_fig_theta = plt.figure(figsize=(x_size*1.2, x_size*1.2),
                                # constrained_layout=True,
                                dpi=dpi_value,
                                )
        grid = corner_fig_theta.add_gridspec(
            nrows=3, ncols=3,  # width_ratios=[3, 1, 1],
        )

        corner.corner(np.array(theta_samples), levels=(0.68, 0.95), labels=var_label,
                                   truths=np.array(theta_true), truth_color='red', show_titles=True,
                                   quantiles=(0.16, 0.5, 0.84),
                                   data_kwargs={'ms': 1.0, 'alpha': 0.1}, hist_kwargs=dict(density=True),fig=corner_fig_theta)
        corner_fig_theta.text(0.5, 0.8, f"true theta: {np.array2string(theta_true, precision=2, floatmode='fixed')}",{'fontsize': 5, 'color': 'red'})
        '''
        x_true = self.theta_to_x(theta_true)
        corner_fig_x = corner.corner(np.array(x_samples), levels=(0.68, 0.95), color='purple', labels=var_label,
                                     truths=np.array(x_true), truth_color='red', show_titles=True,
                                     quantiles=(0.16, 0.5, 0.84), title_kwargs={"fontsize": 15},
                                     label_kwargs={'fontsize': 15},
                                     data_kwargs={'ms': 1.0, 'alpha': 0.1}, hist_kwargs=dict(density=True))
        corner_fig_x.text(0.5, 0.8, f'true x:{x_true}')
        '''
        if save_bool:
            if os.path.exists(f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{z_string}/hmc/') is False:
                os.makedirs(f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{z_string}/hmc/')
            corner_fig_theta.savefig(f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{z_string}/hmc/corner_theta_T{closest_temp_idx}_G{closest_gamma_idx}_F{closest_fobs_idx}_{save_str}.pdf')
            #corner_fig_x.savefig(f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{z_string}/hmc/corner_x_T{closest_temp_idx}_G{closest_gamma_idx}_F{closest_fobs_idx}_{save_str}.pdf')
            print(f"corner plots saved at /mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{z_string}/hmc")
        else:
            return corner_fig_theta

    def fit_plot(self,z_string,theta_samples,lnP,theta_true,model_corr,mock_corr,covariance,save_bool=False,save_str=None):
        '''
        Plot the fit for the HMC results
        Parameters
        ----------
        z_string: str of redshift ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
        theta_samples: list of samples from HMC in theta space, shape (nwalkers * nsteps, ndim)
        lnP: list of log likelihood from HMC, shape (nwalkers * nsteps,)
        theta_true: list of true values of theta [fob, T0, gamma], shape (ndim,)
        model_corr: true model correlation function ['mean_data']
        mock_corr: true mock correlation function for inference
        covariance: model-dependent covariant matrix
        save_bool: True for save at /mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{z_string}/hmc/
        save_str: 'ngp_hmc_test' for NGP model; None for emulator

        Returns
        -------
        fit_fig: fit plot

        '''
        f_mcmc, t_mcmc, g_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                     zip(*np.percentile(theta_samples, [16, 50, 84], axis=0)))
        y_error = np.sqrt(np.diag(covariance))

        fit_fig = plt.figure(figsize=(x_size * 2., x_size * .65), constrained_layout=True,
                             dpi=dpi_value)
        grid = fit_fig.add_gridspec(nrows=1, ncols=1)
        fit_axis = fit_fig.add_subplot(grid[0])
        inds = np.random.randint(len(theta_samples), size=100)
        for idx, ind in enumerate(inds):
            sample = theta_samples[ind]
            model_plot = self.get_model_nearest_fine(sample)
            if idx == 0:
                fit_axis.plot(self.vbins, model_plot, c="b", lw=.7, alpha=0.12, zorder=1, label='Posterior Draws')
            else:
                fit_axis.plot(self.vbins, model_plot, c="b", lw=.7, alpha=0.12, zorder=1)
        max_P = max(lnP)
        max_P_idx = [index for index, item in enumerate(lnP) if item == max_P]
        max_P_model = self.get_model_nearest_fine(theta_samples[max_P_idx, :][0])
        infer_model = self.get_model_nearest_fine([f_mcmc[0], t_mcmc[0], g_mcmc[0]])
        fit_axis.plot(self.vbins, infer_model, c="r", label='Inferred Model', zorder=3, lw=1.75,
                      path_effects=[pe.Stroke(linewidth=1.25, foreground='k'), pe.Normal()])
        fit_axis.plot(self.vbins, model_corr, c="green", ls='--', label='True Model', zorder=5, lw=1.75,
                      path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])
        fit_axis.scatter(self.vbins, mock_corr, c="k", zorder=3, s=2)
        fit_axis.plot(self.vbins, max_P_model, c="gold", label='Max Probability Model', zorder=4, lw=0.75,
                      path_effects=[pe.Stroke(linewidth=1.25, foreground='k'), pe.Normal()])
        fit_axis.errorbar(self.vbins, mock_corr,
                          yerr=y_error,
                          color='k', marker='.', linestyle=' ', zorder=2, capsize=2,
                          label='Mock Data')

        fit_axis.text(
            0.15, 0.6,
            'True Model \n' + r'$\langle F \rangle$' + f' = {np.round(theta_true[0], decimals=4)}' + f'\n $T_0$ = {int(theta_true[1])} K \n $\gamma$ = {np.round(theta_true[2], decimals=3)} \n',
            {'color': 'green', 'fontsize': 5}, transform=fit_axis.transAxes, fontsize=7
        )

        fit_axis.text(
            0.32, 0.55,
            'Inferred Model \n' + r'$\langle F \rangle$' + f' = {np.round(f_mcmc[0], decimals=4)}$^{{+{np.round(f_mcmc[1], decimals=4)}}}_{{-{np.round(f_mcmc[2], decimals=4)}}}$' +
            f'\n $T_0$ = {int(t_mcmc[0])}$^{{+{int(t_mcmc[1])}}}_{{-{int(t_mcmc[2])}}}$ K'
            f'\n ' + r'$\gamma$' + f' = {np.round(g_mcmc[0], decimals=3)}$^{{+{np.round(g_mcmc[1], decimals=3)}}}_{{-{np.round(g_mcmc[2], decimals=3)}}}$\n',
            {'color': 'r', 'fontsize': 5}, transform=fit_axis.transAxes, fontsize=7
        )

        fit_axis.text(
            0.55, 0.65,
            tabulate([[r' $R_2$',
                       np.round(r2_score(model_corr, infer_model), decimals=4)],
                      ['1-MAPE',
                       np.round(1-mean_absolute_percentage_error(model_corr, infer_model), decimals=4)],
                      ],
                     headers=['Matrics', 'Values'], tablefmt='orgtbl'),
            {'color': 'm', 'fontsize': 7}, transform=fit_axis.transAxes,  fontsize=7
        )
        fit_axis.set_xlim(self.vbins[0], self.vbins[-1])
        fit_axis.set_xlabel("Velocity (km/s)")
        fit_axis.set_ylabel(r"$\xi_F$")
        fit_axis.legend()
        if save_bool:
            if os.path.exists(f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{z_string}/hmc/') is False:
                os.makedirs(f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{z_string}/hmc/')
            closest_temp_idx = np.argmin(np.abs(self.T0s - theta_true[1]))
            closest_gamma_idx = np.argmin(np.abs(self.gammas - theta_true[2]))
            closest_fobs_idx = np.argmin(np.abs(self.fobs - theta_true[0]))
            fit_fig.savefig(f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{z_string}/hmc/fit_plot_theta_T{closest_temp_idx}_G{closest_gamma_idx}_F{closest_fobs_idx}_{save_str}.pdf')
            print(f"fitting plot saved at /mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{z_string}/hmc")
        else:
            return fit_fig