import sys
import os
import igm_emulator as emu
from nn_hmc_3d_x import NN_HMC_X
import dill
import numpy as np
import IPython
import jax
import jax.random as random, uniform, PRNGKey
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
from progressbar import ProgressBar

sys.path.append(os.path.expanduser('~') + '/dw_inference/dw_inference/inference')
sys.path.append(os.path.expanduser('~') + '/wdm/correlation/')


'''
load model and auto-corr
'''
redshift = 5.4
n_inference = 100
n_params = 3

# set emulator parameters
loss_str = 'mse'#'chi_one_covariance'
l2 = 0.0001
activation= jax.nn.leaky_relu

# get the appropriate string and pathlength for chosen redshift
zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
z_idx = np.argmin(np.abs(zs - redshift))
z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
z_string = z_strings[z_idx]
if emu.small_bin_bool == True:
    n_path = 20  # 17->20
    n_covar = 500000
    bin_label = '_set_bins_3'
    in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final_135/{z_string}/'
    out_tag = f'{z_string}_training_768_bin59'
    output_size = [100, 100, 100, 59]
else:
    # n_paths = np.array([17, 16, 16, 15, 15, 15, 14]) #skewers_per_data
    # n_path = n_paths[z_idx]
    n_path = 17
    n_covar = 500000
    bin_label = '_set_bins_4'
    in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'
    out_tag = f'{z_string}_training_768_bin276'
    output_size = [100, 100, 100, 276]

# load model
in_path_hdf5 = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/best_params/'
var_tag = f'{loss_str}_l2_{l2}_activation_{activation.__name__}_layers_{output_size}'
best_params = dill.load(open(in_path_hdf5 + f'{out_tag}_{var_tag}_best_param.p', 'rb'))

in_name_h5py = f'correlation_temp_fluct_skewers_2000_R_30000_nf_9_dict{bin_label}.hdf5'
with h5py.File(in_path + in_name_h5py, 'r') as f:
    params = dict(f['params'].attrs.items())
fobs = params['average_observed_flux']
R_value = params['R']
vbins = params['v_bins']
T0s = 10. ** params['logT_0']
gammas = params['gamma']
n_f = len(fobs)
n_temps = len(T0s)
n_gammas = len(gammas)

noise_idx = 0
T0_idx = 8
g_idx = 4
f_idx = 4
like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{g_idx}_SNR0_F{f_idx}_ncovar_{n_covar}_P{n_path}{bin_label}.p'
like_dict = dill.load(open(in_path + like_name, 'rb'))

# get priors
nn_x = NN_HMC_X(vbins, best_params, T0s, gammas, fobs, like_dict)

# get 1000 sampled parameters
true_theta_sampled = np.empty([n_inference, n_params])
rng = np.random.default_rng(36)
seed = rng.choice(100,3)

true_temp_x = uniform(PRNGKey(seed[0]),n_inference, minval=nn_x.theta_to_x(T0s)[0], maxval=nn_x.theta_to_x(T0s)[-1])

true_gamma_x = uniform(PRNGKey(seed[1]),n_inference, minval=nn_x.theta_to_x(gammas)[0], maxval=nn_x.theta_to_x(gammas)[-1])

true_fobs_x = uniform(PRNGKey(seed[2]),n_inference, minval=nn_x.theta_to_x(fobs)[0], maxval=nn_x.theta_to_x(fobs)[-1])

true_theta_sampled[:, 0] = nn_x.x_to_theta(true_temp_x)
true_theta_sampled[:, 1] = nn_x.x_to_theta(true_gamma_x)
true_theta_sampled[:, 2] = nn_x.x_to_theta(true_fobs_x)


'''
in_path_model = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'
molly_name = f'z54_data_nearest_model_set_bins_4_steps_48000_mcmc_inference_5_one_prior_T{T0_idx}_G{g_idx}_F{f_idx}_R_30000.hdf5'
molly_model = h5py.File(in_path_model + molly_name, 'r')
'''


'''
Run HMC
'''
if __name__ == '__main__':
    note = 'gaussian_emulator_prior_x'
    save_name = f"{out_tag}_inference_{n_inference}_{note}_samples_{nn_x.num_samples}_chains_{nn_x.num_chains}_{var_tag}"

    key = random.PRNGKey(642)
    key, subkey = random.split(key)

    true_theta = np.empty([n_inference, n_params])
    infer_theta = np.empty([n_inference, n_params])
    log_prob = np.empty([n_inference, nn_x.num_samples*nn_x.num_chains])
    true_log_prob = np.empty([n_inference])
    samples = np.empty([n_inference, nn_x.num_samples*nn_x.num_chains, n_params])

    var_label = ['fobs', 'T0s', 'gammas']
    out_path = '/mnt/quasar2/zhenyujin/igm_emulator/hmc/hmc_results/'
    pbar = ProgressBar()
    print(f'Start {n_inference} inference test for:{save_name}')
    for mock_idx in pbar(range(n_inference)):
        closest_temp_idx = np.argmin(np.abs(T0s - true_theta_sampled[mock_idx, 0]))
        closest_gamma_idx = np.argmin(np.abs(gammas - true_theta_sampled[mock_idx, 1]))
        closest_fobs_idx = np.argmin(np.abs(fobs - true_theta_sampled[mock_idx, 2]))
        true_theta[mock_idx, :] = [fobs[closest_fobs_idx], T0s[closest_temp_idx], gammas[closest_gamma_idx]]
        x_true = nn_x.theta_to_x(true_theta[mock_idx, :])
        #mock_name = f'mocks_R_{int(R_value)}_nf_{n_f}_T{closest_temp_idx}_G{closest_gamma_idx}_SNR{noise_idx}_F{closest_fobs_idx}_P{n_path}{bin_label}.p'
        #mocks = dill.load(open(in_path + mock_name, 'rb'))
        #flux = mocks[mock_idx, :]
        mock_name = f'{note}_corr_inference{n_inference}_{var_tag}.p'
        mocks = dill.load(open(out_path + mock_name, 'rb'))
        flux = mocks[mock_idx, :]

        x_samples, theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean, \
        hmc_num_steps, hmc_tree_depth, total_time = nn_x.mcmc_one(key, x_true, flux, report=False)
        f_mcmc, t_mcmc, g_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                     zip(*np.percentile(theta_samples, [16, 50, 84], axis=0)))

        infer_theta[mock_idx, :] = [f_mcmc[0], t_mcmc[0], g_mcmc[0]]
        samples[mock_idx, :, :] = theta_samples
        log_prob[mock_idx, :] = lnP
        true_log_prob[mock_idx] = nn_x.potential_fun(x_true, flux)
        #corner plot for each inference
        if mock_idx < 10:
            corner_fig = corner.corner(np.array(theta_samples), levels=(0.68, 0.95), labels=var_label,
                                       truths=np.array(true_theta[mock_idx, :]), truth_color='red', show_titles=True,
                                       quantiles=(0.16, 0.5, 0.84),title_kwargs={"fontsize": 15}, label_kwargs={'fontsize': 20},
                                       data_kwargs={'ms': 1.0, 'alpha': 0.1}, hist_kwargs=dict(density=True))
            corner_fig.savefig(f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{z_string}/corner_T{closest_temp_idx}_G{closest_gamma_idx}_SNR{noise_idx}_F{closest_fobs_idx}_P{n_path}{bin_label}_mock_{mock_idx}_small_bins_{note}.png')

        #save HMC inference results
        with h5py.File(out_path + f'{save_name}.hdf5', 'a') as f:
            f.create_dataset('true_theta', data=true_theta)
            f.create_dataset('log_prob', data=log_prob)
            f.create_dataset('true_log_prob', data=true_log_prob)
            f.create_dataset('samples', data=samples)
            f.create_dataset('infer_theta', data=infer_theta)
        IPython.embed()

