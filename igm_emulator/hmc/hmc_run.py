from hmc_nn_inference import NN_HMC_X
import dill
import numpy as np
import IPython
import jax.random as random
from sklearn.metrics import mean_squared_error,r2_score
from scipy.spatial.distance import minkowski
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as pe
from tabulate import tabulate
import corner
import h5py
import sys
import os
from igm_emulator.emulator.emulator_apply import trainer, small_bin_bool, test_num, z_string
print(f'Training for small bin: {small_bin_bool}')
from progressbar import ProgressBar
#sys.path.append(os.path.expanduser('~') + '/dw_inference/dw_inference/inference')
#sys.path.append(os.path.expanduser('~') + '/wdm/correlation/')
#from mcmc_inference_new_linda_params_mult_file_3d import return_idx, get_model_covar_nearest

'''
load params and auto-corr
'''
compare = False

T0_idx = 7 #0-14
g_idx = 4 #0-8
f_idx = 4 #0-8


if small_bin_bool==True:
    n_path = 20  # 17->20
    n_covar = 500000
    bin_label = '_set_bins_3'
    in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final_135/{z_string}/'
else:
    #n_paths = np.array([17, 16, 16, 15, 15, 15, 14]) #skewers_per_data
    #n_path = n_paths[z_idx]
    n_path = 17
    n_covar = 500000
    bin_label = '_set_bins_4'
    in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'

in_name_h5py = f'correlation_temp_fluct_skewers_2000_R_30000_nf_9_dict{bin_label}.hdf5'
with h5py.File(in_path + in_name_h5py, 'r') as f:
    params = dict(f['params'].attrs.items())
fobs = params['average_observed_flux']
R_value = params['R']
vbins = params['v_bins']
T0s = 10. ** params['logT_0']
v_bins = params['v_bins']
R_value = params['R']
gammas = params['gamma']
n_f = len(fobs)

noise_idx = 0
like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{g_idx}_SNR0_F{f_idx}_ncovar_{n_covar}_P{n_path}{bin_label}.p'
like_dict = dill.load(open(in_path + like_name, 'rb'))
mock_name = f'mocks_R_{int(R_value)}_nf_{n_f}_T{T0_idx}_G{g_idx}_SNR{noise_idx}_F{f_idx}_P{n_path}{bin_label}.p'
mocks = dill.load(open(in_path + mock_name, 'rb'))
theta_true = np.array([fobs[f_idx], T0s[T0_idx], gammas[g_idx]])
print(f'true theta:{theta_true}')
mock_flux = mocks[0:10,:]
mean_flux = like_dict['mean_data']
new_covariance = like_dict['covariance']

'''
load emulator and error approx
'''
in_path_best_params = '/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/hparam_results/'
in_path_hdf5 = '/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/'
var_tag = trainer.var_tag
out_tag = trainer.out_tag
best_params = dill.load(open(in_path_best_params + f'{out_tag}_{var_tag}_best_param.p',
                                  'rb'))
covar_nn = dill.load(open(in_path_hdf5 + f'{out_tag}{test_num}_{var_tag}_covar_nn.p', 'rb'))
err_nn = dill.load(open(in_path_hdf5 + f'{out_tag}{test_num}_{var_tag}_err_nn.p', 'rb'))

'''
Run HMC
'''
if __name__ == '__main__':
    nn_x = NN_HMC_X(v_bins, best_params,T0s, gammas, fobs,  #switch to new_temps, new_gammas, new_fobs didn't change anything
                                dense_mass=True,
                                max_tree_depth= 10,
                                num_warmup=1000,
                                num_samples=1000,
                                num_chains=4,
                                covar_nn=covar_nn,
                                err_nn=err_nn,
                                nn_err_prop =True)
    key = random.PRNGKey(642)
    key, subkey = random.split(key)
    var_label = ['fobs', 'T0s', 'gammas']
    n_inference = 5
    idx = np.random.randint(10, size=n_inference)
    pbar = ProgressBar()
    for mock_idx in pbar(idx):
        flux = mocks[mock_idx, :]
        x_opt, theta_opt, losses = nn_x.fit_one(flux, new_covariance)
        x_samples, theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean, \
            hmc_num_steps, hmc_tree_depth, total_time = nn_x.mcmc_one(subkey, x_opt, flux, new_covariance, report=True)
        f_mcmc, t_mcmc, g_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                     zip(*np.percentile(theta_samples, [16, 50, 84], axis=0)))
        nn_x.save_HMC(z_string,f_idx, T0_idx,g_idx, f_mcmc, t_mcmc, g_mcmc, x_samples, theta_samples,theta_true, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean,
                 hmc_num_steps, hmc_tree_depth, total_time,save_str=f'central_mock_{mock_idx}')

        corner_fig = nn_x.corner_plot(z_string, theta_samples, x_samples, theta_true,
                                        save_str=f'central_mock_{mock_idx}', save_bool=True)
        fit_fig = nn_x.fit_plot(z_string=z_string, theta_samples=theta_samples, lnP=lnP,
                                   theta_true=theta_true, model_corr=mean_flux,
                                   mock_corr=flux,
                                   covariance=new_covariance, save_str=f'central_mock_{mock_idx}', save_bool=True)

    flux = mean_flux
    x_opt, theta_opt, losses = nn_x.fit_one(flux, new_covariance)
    x_samples, theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean, \
        hmc_num_steps, hmc_tree_depth, total_time = nn_x.mcmc_one(subkey, x_opt, flux, new_covariance, report=True)
    f_mcmc, t_mcmc, g_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                 zip(*np.percentile(theta_samples, [16, 50, 84], axis=0)))
    nn_x.save_HMC(z_string,f_idx, T0_idx,g_idx, f_mcmc, t_mcmc, g_mcmc, x_samples, theta_samples,theta_true, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean,
             hmc_num_steps, hmc_tree_depth, total_time,save_str=f'central_mean_model')

    corner_fig = nn_x.corner_plot(z_string, theta_samples, x_samples, theta_true,
                                    save_str=f'central_mean_model', save_bool=True)
    fit_fig = nn_x.fit_plot(z_string=z_string, theta_samples=theta_samples, lnP=lnP,
                               theta_true=theta_true, model_corr=mean_flux,
                               mock_corr=flux,
                               covariance=new_covariance, save_str=f'central_mean_model', save_bool=True)
        #if compare:
        #    in_path_model = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'
        #    molly_name = f'z54_data_nearest_model_set_bins_4_steps_48000_mcmc_inference_5_one_prior_T{T0_idx}_G{g_idx}_F{f_idx}_R_30000.hdf5'
        #    molly_model = h5py.File(in_path_model + molly_name, 'r')
        #
        #    molly_sample = molly_model['samples'][mock_idx, :, :]
        #    molly_flip = np.zeros(shape=molly_sample.shape)
        #    molly_flip[:, 0] = molly_sample[:, 2]
        #    molly_flip[:, 1] = molly_sample[:, 0]
        #    molly_flip[:, 2] = molly_sample[:, 1]
        #
        #    t_molly, g_molly, f_molly = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
        #                                    zip(*np.percentile(molly_sample, [16, 50, 84], axis=0)))
        #    molly_infer, covar, log_det = get_model_covar_nearest([t_molly[0], g_molly[0], f_molly[0]])
        #
        #    corner.corner(molly_flip, levels=(0.68, 0.95), fig=corner_fig, color='blue',hist_kwargs=dict(density=True))
