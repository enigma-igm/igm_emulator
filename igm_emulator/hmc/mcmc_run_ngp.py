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
from IPython import embed
from nn_hmc_3d_x import NN_HMC_X
class HMC_NGP(NN_HMC_X):
    def __init__(self, vbins, T0s, gammas, fobs, fine_models, fine_covariances, fine_log_dets,
                dense_mass=True,
                 max_tree_depth= 10, #(8,10),
                 num_warmup=1000,
                 num_samples=1000,
                 num_chains=4):
        super().__init__(vbins=vbins,
                         best_params=None,
                         T0s=T0s,
                         gammas=gammas,
                         fobs=fobs,
                         like_dict=None,
                         dense_mass=dense_mass,
                         max_tree_depth= max_tree_depth,
                         num_warmup=num_warmup,
                         num_samples=num_samples,
                         num_chains=num_chains)

        self.fine_models = fine_models
        self.fine_covariances = fine_covariances
        self.fine_log_dets = fine_log_dets
    @partial(jit, static_argnums=(0,))
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
    embed()

    hmc_ngp = HMC_NGP(v_bins, new_temps, new_gammas, new_fobs, new_models, new_covariances, new_log_dets)
    flux = hmc_ngp.get_model_nearest_fine(theta_true)
    x_true = hmc_ngp.theta_to_x(theta_true)
    cov, log_det = hmc_ngp.get_covariance_log_determinant_nearest_fine(theta_true)
    theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean, hmc_num_steps, hmc_tree_depth, total_time =  hmc_ngp.mcmc_one(key, x_true, flux, cov, report=True)
