from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np
import os
import jax.numpy as jnp
import jax.numpy as jnp
import jax
from jax.scipy.stats import multivariate_normal
from jax import jit
import jax.random as random
from sklearn.metrics import mean_squared_error,r2_score
from scipy.spatial.distance import minkowski
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
from hmc_nn_inference import NN_HMC_X
from igm_emulator.emulator.emulator_apply import trainer
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

        closest_temp_idx = jnp.argmin(jnp.abs(self.T0s - T0))
        closest_gamma_idx = jnp.argmin(jnp.abs(self.gammas - gamma))
        closest_fobs_idx = jnp.argmin(jnp.abs(self.fobs - fob))


        covariance = self.fine_covariances[closest_temp_idx, closest_gamma_idx, closest_fobs_idx, :, :]
        log_determinant = self.fine_log_dets[closest_temp_idx, closest_gamma_idx, closest_fobs_idx]

        return covariance, log_determinant

    @partial(jit, static_argnums=(0,))
    def get_model_nearest_fine(
            self, theta
    ):

        fob, T0, gamma = theta

        closest_temp_idx = jnp.argmin(jnp.abs(self.T0s - T0))
        closest_gamma_idx = jnp.argmin(jnp.abs(self.gammas - gamma))
        closest_fobs_idx = jnp.argmin(jnp.abs(self.fobs - fob))

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
    n_path = 20
    v_bins = params['v_bins']
    t_0s = 10.**params['logT_0']
    gammas = params['gamma']
    n_temps = len(t_0s)
    n_gammas = len(gammas)
    n_f = len(fobs)
    #
    # noise_idx = 0
    # in_name_new_params = f'new_covariances_dict_R_30000_nf_9_ncovar_{n_covar}_' \
    #                      f'P{skewers_per_data}{bin_label}_params.p'
    # new_param_dict = dill.load(open(in_path_molly + in_name_new_params, 'rb'))
    # new_temps = new_param_dict['new_temps']
    # new_gammas = new_param_dict['new_gammas']
    # new_fobs = new_param_dict['new_fobs']
    #
    # n_new_t = (len(new_temps) - 1)/(len(t_0s) - 1) - 1
    # n_new_g = (len(new_gammas) - 1)/(len(gammas) - 1) - 1
    # n_new_f = (len(new_fobs) - 1)/(len(fobs) - 1) - 1
    # new_models_np = np.empty([len(new_temps), len(new_gammas), len(new_fobs), len(v_bins)])
    # new_covariances_np = np.empty([len(new_temps), len(new_gammas), len(new_fobs), len(v_bins),len(v_bins)])
    # new_log_dets_np = np.empty([len(new_temps), len(new_gammas), len(new_fobs)])
    #
    #
    # for old_t_below_idx in range(n_temps - 1):
    #     print(f'{old_t_below_idx / (n_temps - 1) * 100}%')
    #     for old_g_below_idx in range(n_gammas - 1):
    #         for old_f_below_idx in range(n_f - 1):
    #             fine_dict_in_name = f'new_covariances_dict_R_{int(R_value)}_nf_{n_f}_T{old_t_below_idx}_' \
    #                                 f'G{old_g_below_idx}_SNR0_F{old_f_below_idx}_ncovar_{n_covar}_' \
    #                                 f'P{skewers_per_data}{bin_label}.p'
    #             fine_dict = dill.load(open(in_path_molly + fine_dict_in_name, 'rb'))
    #             new_temps_small = fine_dict['new_temps']
    #             new_gammas_small = fine_dict['new_gammas']
    #             new_fobs_small = fine_dict['new_fobs']
    #             new_models_small = fine_dict['new_models']
    #             new_covariances_small = fine_dict['new_covariances']
    #             new_log_dets_small = fine_dict['new_log_dets']
    #             if old_t_below_idx == n_temps - 2:
    #                 added_t_range = n_new_t + 2
    #             else:
    #                 added_t_range = n_new_t + 1
    #             if old_g_below_idx == n_gammas - 2:
    #                 added_g_range = n_new_g + 2
    #             else:
    #                 added_g_range = n_new_g + 1
    #             if old_f_below_idx == n_f - 2:
    #                 added_f_range = n_new_f + 2
    #             else:
    #                 added_f_range = n_new_f + 1
    #             # print(added_f_range)
    #             for added_t_idx in range(int(added_t_range)):
    #                 for added_g_idx in range(int(added_g_range)):
    #                     for added_f_idx in range(int(added_f_range)):
    #                         final_t_idx = int((old_t_below_idx * (n_new_t + 1)) + added_t_idx)
    #                         final_g_idx = int((old_g_below_idx * (n_new_g + 1)) + added_g_idx)
    #                         final_f_idx = int((old_f_below_idx * (n_new_f + 1)) + added_f_idx)
    #                         new_models_np[final_t_idx, final_g_idx, final_f_idx, :] = new_models_small[added_t_idx, added_g_idx, added_f_idx, :]
    #                         new_covariances_np[final_t_idx, final_g_idx, final_f_idx, :, :] = new_covariances_small[added_t_idx, added_g_idx, added_f_idx, :, :]
    #                         new_log_dets_np[final_t_idx, final_g_idx, final_f_idx] = new_log_dets_small[added_t_idx, added_g_idx, added_f_idx]
    # new_models = jnp.array(new_models_np)
    # new_covariances = jnp.array(new_covariances_np)
    # new_log_dets = jnp.array(new_log_dets_np)

    T0_idx = 0  # 0-14
    g_idx = 5  # 0-8
    f_idx = 4  # 0-8
    theta_true = [fobs[f_idx], t_0s[T0_idx], gammas[g_idx]]
    mock_name = f'mocks_R_{int(R_value)}_nf_{n_f}_T{T0_idx}_G{g_idx}_SNR{noise_idx}_F{f_idx}_P{n_path}{bin_label}.p'
    model_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{g_idx}_SNR0_F{f_idx}_ncovar_{n_covar}_P{n_path}{bin_label}.p'
    model_dict = dill.load(open(in_path_molly + model_name, 'rb'))
    model = model_dict['mean_data']
    mocks = dill.load(open(in_path_molly + mock_name, 'rb'))
    #embed()
    in_path_hdf5 = '/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/hparam_results/'
    best_params = dill.load(
        open(in_path_hdf5 + f'{trainer.out_tag}_{trainer.var_tag}_best_param.p', 'rb'))  # changed to optuna tuned best param

    flux = mocks[0, :]
    key = random.PRNGKey(642)
    print(theta_true)

    '''
    NGP HMC 
    '''
    # # Define the NGP inference
    # hmc_ngp = HMC_NGP(v_bins, new_temps, new_gammas, new_fobs, new_models, new_covariances, new_log_dets)
    # cov, log_det = hmc_ngp.get_covariance_log_determinant_nearest_fine(theta_true)
    # x_out, theta_out, losses = hmc_ngp.fit_one(flux, cov)
    # print(f'NGP [fobs, T0, gamma] ranges: {hmc_ngp.theta_ranges}')
    # print(np.argmin(np.abs(hmc_ngp.T0s - theta_true[1])),
    #       np.argmin(np.abs(hmc_ngp.gammas - theta_true[2])),
    #       np.argmin(np.abs(hmc_ngp.fobs - theta_true[0])))
    #
    # key, subkey = random.split(key)
    # x_samples_ngp, theta_samples_ngp, lnP_ngp, neff_ngp, neff_mean_ngp, sec_per_neff_ngp, ms_per_step_ngp, r_hat_ngp, r_hat_mean_ngp, \
    #     hmc_num_steps_ngp, hmc_tree_depth_ngp, total_time_ngp = hmc_ngp.mcmc_one(subkey, x_out, flux, cov, report=True)
    # hmc_ngp.corner_plot(zstr, theta_samples_ngp, x_samples_ngp, theta_out, save_str='ngp_hmc_test')
    # hmc_ngp.fit_plot(zstr, theta_samples_ngp, lnP_ngp, theta_out, model_corr=model, mock_corr=flux, covariance=cov,
    #                  save_bool=True, save_str='ngp_hmc_test')
    # _, _, _, _, logP_grid_ngp, chi_grid_ngp = hmc_ngp.explore_logP_plot(zstr, theta_true=theta_out, flux=flux,
    #                                                                     covar=cov, fix='t', save_str='ngp_hmc_test')

    '''
    Emulator HMC
    '''
    #Define the NN inference
    hmc_nn = NN_HMC_X(v_bins, best_params, t_0s, gammas, fobs, dense_mass=True,
                        max_tree_depth= 10,
                        num_warmup=1000,
                        num_samples=1000,
                        num_chains=4)
    x_out, theta_out, losses = hmc_nn.fit_one(flux, cov)
    print(f'NN [fobs, T0, gamma] ranges: {hmc_nn.theta_ranges}')
    closest_temp_idx = np.argmin(np.abs(hmc_nn.T0s - theta_true[1]))
    closest_gamma_idx = np.argmin(np.abs(hmc_nn.gammas - theta_true[2]))
    closest_fobs_idx = np.argmin(np.abs(hmc_nn.fobs - theta_true[0]))

    key, subkey = random.split(key)
    x_samples, theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean, \
        hmc_num_steps, hmc_tree_depth, total_time = hmc_nn.mcmc_one(subkey, x_out, flux, cov, report=True)
    hmc_nn.corner_plot(zstr, theta_samples, x_samples, theta_out, save_str=None, save_bool=True)
    hmc_nn.fit_plot(zstr, theta_samples, lnP, theta_out, model_corr=model, mock_corr=flux, covariance=cov,
                    save_bool=True, save_str=None)
    fix, f_grid, t_grid, g_grid, logP_grid_nn, chi_grid_nn = hmc_nn.explore_logP_plot(zstr, theta_true=theta_out,
                                                                                      flux=flux, covar=cov, fix='t')

    '''
    Plot NGP and NN lnP and chi difference
    '''
    # plt.figure(figsize=(10, 8))
    # plt.imshow(logP_grid_nn-logP_grid_ngp, extent=[f_grid.min(), f_grid.max(), g_grid.min(), g_grid.max()], origin='lower',
    #            aspect='auto', cmap='viridis',vmin=0, vmax=25)
    # plt.colorbar(label='lnP NN-NGP')
    # plt.xlabel('f_grid')
    # plt.ylabel('g_grid')
    # plt.title('Color plot of lnP NN-NGP')
    # plt.savefig(f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{zstr}/hmc/lnP_subtract_fix_{fix}_T{closest_temp_idx}_G{closest_gamma_idx}_F{closest_fobs_idx}.pdf')
    # plt.close()
    #
    # plt.figure(figsize=(10, 8))
    # plt.imshow(chi_grid_ngp - chi_grid_nn, extent=[f_grid.min(), f_grid.max(), g_grid.min(), g_grid.max()],
    #            origin='lower',
    #            aspect='auto', cmap='viridis',vmin=0, vmax=0.3)
    # plt.colorbar(label='Chi NGP-NN')
    # plt.xlabel('f_grid')
    # plt.ylabel('g_grid')
    # plt.title('Color plot of Chi NGP-NN')
    # plt.savefig(
    #     f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{zstr}/hmc/chi_subtract_fix_{fix}_T{closest_temp_idx}_G{closest_gamma_idx}_F{closest_fobs_idx}.pdf')
    # plt.close()