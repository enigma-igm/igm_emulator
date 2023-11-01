from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import jax.numpy as jnp
import jax
from jax.scipy.stats import multivariate_normal
from jax import jit
import jax.random as random
from functools import partial
from numpyro.infer import MCMC, NUTS
import h5py
import arviz as az
import time
import dill
import corner
import matplotlib
import matplotlib.pyplot as plt

class HMC_NGP:
    def __init__(self, vbins, T0s, gammas, fobs, fine_models, fine_covariances, fine_log_dets,
                dense_mass=True,
                 max_tree_depth= 10, #(8,10),
                 num_warmup=1000,
                 num_samples=1000,
                 num_chains=4):

        self.vbins = vbins
        self.max_tree_depth = max_tree_depth
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.dense_mass = dense_mass
        self.mcmc_nsteps_tot = num_samples * num_chains
        self.num_samples = num_samples
        self.T0s = T0s
        self.gammas = gammas
        self.fobs = fobs
        self.theta_ranges = [[self.fobs[0], self.fobs[-1]], [self.T0s[0], self.T0s[-1]],
                             [self.gammas[0], self.gammas[-1]]]
        self.fine_models = fine_models
        self.fine_covariances = fine_covariances
        self.fine_log_dets = fine_log_dets
    def get_covariance_log_determinant_nearest_fine(
            self, theta
    ):

        fob, T0, gamma = theta

        closest_temp_idx = np.argmin(np.abs(self.T0s - T0))
        closest_gamma_idx = np.argmin(np.abs(self.gammas - gamma))
        closest_fobs_idx = np.argmin(np.abs(self.fobs - fob))


        covariance = self.fine_covariances[closest_temp_idx, closest_gamma_idx, closest_fobs_idx, :, :]
        log_determinant = self.fine_log_dets[closest_temp_idx, closest_gamma_idx, closest_fobs_idx]

        return covariance, log_determinant


    def get_model_nearest_fine(
            self, theta
    ):

        fob, T0, gamma = theta

        closest_temp_idx = np.argmin(np.abs(self.T0s - T0))
        closest_gamma_idx = np.argmin(np.abs(self.gammas - gamma))
        closest_fobs_idx = np.argmin(np.abs(self.fobs - fob))

        model = self.fine_models[closest_temp_idx, closest_gamma_idx, closest_fobs_idx, :]

        return model


    def log_likelihood(
            self, theta, data_auto_correlation,
            alpha_const=None,
            fixed_covariance_bool=False,
            fixed_covariance=None,
    ):

        model = self.get_model_nearest_fine(theta)

        if fixed_covariance_bool:
            covariance = fixed_covariance
        else:
            covariance, log_determinant = self.get_covariance_log_determinant_nearest_fine(
                theta
            )

        if alpha_const:
            covariance = covariance * alpha_const ** -1

        log_like = multivariate_normal.logpdf(data_auto_correlation, mean=model, cov=covariance)

        return log_like

    def log_prior(self, theta):

        fob, T0, gamma = theta

        if self.fobs[0]<= fob <= self.fobs[1] and self.T0s[0]<= T0 <= self.T0s[1] and self.gammas[0]<=gamma <= self.gammas[1]:
            return 0.0
        return -np.inf

    @partial(jit, static_argnums=(0,))
    def log_probability(
            self, theta, data_auto_correlation,
            alpha_const=None,
            fixed_covariance_bool=False,
            fixed_covariance=None
    ):


        lp = self.log_prior(theta)

        if not np.isfinite(lp):
            return -np.inf

        return lp + self.log_likelihood(
            theta, data_auto_correlation,
            alpha_const=alpha_const,
            fixed_covariance_bool=fixed_covariance_bool,
            fixed_covariance=fixed_covariance,
        )

    @partial(jit, static_argnums=(0,))
    def numpyro_potential_fun(self, corr): #potential function for numpyro
        return jax.tree_util.Partial(self.log_probability, data_auto_correlation=corr)

    def mcmc_one(self, key, theta, corr, report=True):  # input dimensionless paramter x
        # Instantiate the NUTS kernel and the mcmc object
        nuts_kernel = NUTS(potential_fn=self.numpyro_potential_fun(corr),
                           adapt_step_size=True, dense_mass=True, max_tree_depth=self.max_tree_depth)
        # Initial position
        if report:
            mcmc = MCMC(nuts_kernel, num_warmup=self.num_warmup, num_samples=self.num_samples,
                        num_chains=self.num_chains,
                        jit_model_args=True,
                        chain_method='vectorized')  # chain_method='sequential' chain_method='vectorized'
        else:
            mcmc = MCMC(nuts_kernel, num_warmup=self.num_warmup, num_samples=self.num_samples,
                        num_chains=self.num_chains,
                        jit_model_args=True,
                        chain_method='vectorized',
                        progress_bar=False)  # chain_method='sequential' chain_method='vectorized'
        theta_init = theta + 1e-4 * np.random.randn(self.num_chains, 3)

        # Run the MCMC
        start_time = time.time()
        # IPython.embed()
        mcmc.run(key, init_params=theta_init.squeeze(), extra_fields=('potential_energy', 'num_steps'))
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
        #theta_samples = self.x_to_theta(mcmc.get_samples(group_by_chain=False))  # (mcmc_nsteps_tot, ndim)
        theta_samples = mcmc.get_samples(group_by_chain=True)  # (num_chain, num_samples, ndim)
        lnP = -mcmc.get_extra_fields()['potential_energy']
        hmc_num_steps = mcmc.get_extra_fields()[
            'num_steps']  # Number of steps in the Hamiltonian trajectory (for diagnostics).
        hmc_tree_depth = np.log2(hmc_num_steps).astype(
            int) + 1  # Tree depth of the Hamiltonian trajectory (for diagnostics).
        hit_max_tree_depth = np.sum(
            hmc_tree_depth == self.max_tree_depth)  # Number of transitions that hit the maximum tree depth.
        ms_per_step = 1e3 * total_time / np.sum(hmc_num_steps)

        if report:
            print(f"*** SUMMARY FOR HMC ***")
            print(f"total_time = {total_time} seconds for the HMC")
            print(f"total_steps = {np.sum(hmc_num_steps)} total steps")
            print(f"ms_per_step = {ms_per_step} ms per step of the HMC")
            print(
                f"n_eff_mean = {neff_mean} effective sample size, compared to ntot = {self.mcmc_nsteps_tot} total samples.")
            print(f"ms_per_neff = {ms_per_neff:.3f} ms per effective sample")
            print(f"r_hat_mean = {r_hat_mean}")
            print(f"max_tree_depth encountered = {hmc_tree_depth.max()} in chain")
            print(f"There were {hit_max_tree_depth} transitions that exceeded the max_tree_depth = {self.max_tree_depth}")
            print("*************************")

        # Return the values needed
        return  theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean, \
            hmc_num_steps, hmc_tree_depth, total_time


    def save_HMC(self, zstr, note, f_idx, T0_idx, g_idx, f_mcmc, t_mcmc, g_mcmc, x_samples, theta_samples, lnP, neff,
                 neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean,
                 hmc_num_steps, hmc_tree_depth, total_time):
        # Save the results
        with h5py.File(os.path.expanduser(
                '~') + f'/igm_emulator/igm_emulator/hmc/hmc_results/{zstr}_F{f_idx}_T0{T0_idx}_G{g_idx}_{note}_hmc_results.hdf5',
                       'w') as f:
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
        print(f"hmc results saved for {note}")


    def corner_plot(self, z_string, theta_samples, theta_true, save_str=None):

        closest_temp_idx = np.argmin(np.abs(self.T0s - theta_true[1]))
        closest_gamma_idx = np.argmin(np.abs(self.gammas - theta_true[2]))
        closest_fobs_idx = np.argmin(np.abs(self.fobs - theta_true[0]))
        var_label = ['fobs', 'T0s', 'gammas']

        corner_fig_theta = corner.corner(np.array(theta_samples), levels=(0.68, 0.95), labels=var_label,
                                         truths=np.array(theta_true), truth_color='red', show_titles=True,
                                         title_kwargs={"fontsize": 9}, label_kwargs={'fontsize': 20},
                                         data_kwargs={'ms': 1.0, 'alpha': 0.1}, hist_kwargs=dict(density=True))
        corner_fig_theta.text(0.5, 0.8, f'true theta:{theta_true}')

        corner_fig_theta.savefig(
            f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{z_string}/corner_theta_T{closest_temp_idx}_G{closest_gamma_idx}_F{closest_fobs_idx}_{save_str}.pdf')

if __name__ == '__main__':
    zstr = 'z54'
    skewers_per_data = 20 #17->20
    n_covar = 500000
    bin_label = '_set_bins_3'
    in_path_molly = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final_135/{zstr}/'
    #change path from f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'

    # get initial grid
    in_name_h5py = f'correlation_temp_fluct_skewers_2000_R_30000_nf_9_dict{bin_label}.hdf5'
    with h5py.File(in_path_molly + in_name_h5py, 'r') as f:
        params = dict(f['params'].attrs.items())
    fobs = params['average_observed_flux']
    R_value = params['R']
    v_bins = params['v_bins']
    t_0s = 10.**params['logT_0']
    gammas = params['gamma']
    n_temps = len(t_0s)
    n_gammas = len(gammas)
    n_f = len(fobs)

    noise_idx = 0
    in_name_new_params = f'new_covariances_dict_R_30000_nf_9_ncovar_{n_covar}_' \
                         f'P{skewers_per_data}{bin_label}_params.p'
    new_param_dict = dill.load(open(in_path_molly + in_name_new_params, 'rb'))
    new_temps = new_param_dict['new_temps']
    new_gammas = new_param_dict['new_gammas']
    new_fobs = new_param_dict['new_fobs']

    n_new_t = (len(new_temps) - 1)/(len(t_0s) - 1) - 1
    n_new_g = (len(new_gammas) - 1)/(len(gammas) - 1) - 1
    n_new_f = (len(new_fobs) - 1)/(len(fobs) - 1) - 1
    new_models_np = np.empty([len(new_temps), len(new_gammas), len(new_fobs), len(v_bins)])
    new_covariances_np = np.empty([len(new_temps), len(new_gammas), len(new_fobs), len(v_bins),len(v_bins)])
    new_log_dets_np = np.empty([len(new_temps), len(new_gammas), len(new_fobs)])


    for old_t_below_idx in range(n_temps - 1):
        print(f'{old_t_below_idx / (n_temps - 1) * 100}%')
        for old_g_below_idx in range(n_gammas - 1):
            for old_f_below_idx in range(n_f - 1):
                fine_dict_in_name = f'new_covariances_dict_R_{int(R_value)}_nf_{n_f}_T{old_t_below_idx}_' \
                                    f'G{old_g_below_idx}_SNR0_F{old_f_below_idx}_ncovar_{n_covar}_' \
                                    f'P{skewers_per_data}{bin_label}.p'
                fine_dict = dill.load(open(in_path_molly + fine_dict_in_name, 'rb'))
                new_temps_small = fine_dict['new_temps']
                new_gammas_small = fine_dict['new_gammas']
                new_fobs_small = fine_dict['new_fobs']
                new_models_small = fine_dict['new_models']
                new_covariances_small = fine_dict['new_covariances']
                new_log_dets_small = fine_dict['new_log_dets']
                if old_t_below_idx == n_temps - 2:
                    added_t_range = n_new_t + 2
                else:
                    added_t_range = n_new_t + 1
                if old_g_below_idx == n_gammas - 2:
                    added_g_range = n_new_g + 2
                else:
                    added_g_range = n_new_g + 1
                if old_f_below_idx == n_f - 2:
                    added_f_range = n_new_f + 2
                else:
                    added_f_range = n_new_f + 1
                # print(added_f_range)
                for added_t_idx in range(int(added_t_range)):
                    for added_g_idx in range(int(added_g_range)):
                        for added_f_idx in range(int(added_f_range)):
                            final_t_idx = int((old_t_below_idx * (n_new_t + 1)) + added_t_idx)
                            final_g_idx = int((old_g_below_idx * (n_new_g + 1)) + added_g_idx)
                            final_f_idx = int((old_f_below_idx * (n_new_f + 1)) + added_f_idx)
                            new_models_np[final_t_idx, final_g_idx, final_f_idx, :] = new_models_small[added_t_idx, added_g_idx, added_f_idx, :]
                            new_covariances_np[final_t_idx, final_g_idx, final_f_idx, :, :] = new_covariances_small[added_t_idx, added_g_idx, added_f_idx, :, :]
                            new_log_dets_np[final_t_idx, final_g_idx, final_f_idx] = new_log_dets_small[added_t_idx, added_g_idx, added_f_idx]
    new_models = jnp.array(new_models_np)
    new_covariances = jnp.array(new_covariances_np)
    new_log_dets = jnp.array(new_log_dets_np)

    key = random.PRNGKey(642)
    key, subkey = random.split(key)
    T0_idx = 11  # 0-14
    g_idx = 4  # 0-8
    f_idx = 7  # 0-8
    theta_true = [fobs[f_idx], t_0s[T0_idx], gammas[g_idx]]

    hmc_ngp = HMC_NGP(v_bins, new_temps, new_gammas, new_fobs, new_models, new_covariances, new_log_dets)
    flux = hmc_ngp.get_model_nearest_fine(theta_true)
    theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean, hmc_num_steps, hmc_tree_depth, total_time =  hmc_ngp.mcmc_one(key, theta_true, flux)
