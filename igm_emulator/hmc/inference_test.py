import sys
import os
import igm_emulator as emu
from nn_hmc_3d_x import NN_HMC_X
import dill
import numpy as np
import IPython
import jax
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
sys.path.append('/home/zhenyujin/qso_fitting/')
import h5py
from qso_fitting.analysis.inf_test import run_inference_test, compute_importance_weights, C_ge, inference_test_plot
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
loss_str = 'mse' #'chi_one_covariance' #'mse'
l2 = 0.0001 #0.01
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



'''
in_path_model = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'
molly_name = f'z54_data_nearest_model_set_bins_4_steps_48000_mcmc_inference_5_one_prior_T{T0_idx}_G{g_idx}_F{f_idx}_R_30000.hdf5'
molly_model = h5py.File(in_path_model + molly_name, 'r')
'''


if __name__ == '__main__':
    nn_x = NN_HMC_X(vbins, best_params, T0s, gammas, fobs, like_dict)

    from sample_mocks import note
    out_path = '/mnt/quasar2/zhenyujin/igm_emulator/hmc/hmc_results/'
    save_name = f"{out_tag}_inference_{n_inference}_{note}_samples_{nn_x.num_samples}_chains_{nn_x.num_chains}_{var_tag}"

    key = random.PRNGKey(642)

    infer_theta = np.empty([n_inference, n_params])
    log_prob = np.empty([n_inference, nn_x.num_samples*nn_x.num_chains])
    true_log_prob = np.empty([n_inference])
    samples = np.empty([n_inference, nn_x.num_samples*nn_x.num_chains, n_params])

    #read in samples
    true_theta = dill.load(open(out_path + f'{note}_theta_inference{n_inference}_{var_tag}.p', 'rb'))
    mock_name = f'{note}_corr_inference{n_inference}_{var_tag}.p'
    mocks = dill.load(open(out_path + mock_name, 'rb'))

    '''
    Run inference test for each mock
    '''
    var_label = ['fobs', 'T0s', 'gammas']
    pbar = ProgressBar()
    print(f'Start {n_inference} inference test for:{save_name}')
    for mock_idx in pbar(range(n_inference)):
        key, subkey = random.split(key)

        closest_temp_idx = np.argmin(np.abs(T0s - true_theta[mock_idx, 1]))
        closest_gamma_idx = np.argmin(np.abs(gammas - true_theta[mock_idx, 2]))
        closest_fobs_idx = np.argmin(np.abs(fobs - true_theta[mock_idx, 0]))

        x_true = nn_x.theta_to_x(true_theta[mock_idx, :])
        flux = mocks[mock_idx, :]

        x_samples, theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean, \
        hmc_num_steps, hmc_tree_depth, total_time = nn_x.mcmc_one(key, x_true, flux, report=False)
        f_mcmc, t_mcmc, g_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                     zip(*np.percentile(theta_samples, [16, 50, 84], axis=0)))

        infer_theta[mock_idx, :] = [f_mcmc[0], t_mcmc[0], g_mcmc[0]]
        samples[mock_idx, :, :] = theta_samples
        log_prob[mock_idx, :] = lnP
        true_log_prob[mock_idx] = -1 * nn_x.potential_fun(x_true, flux)
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
        f.create_dataset('log_prob_x', data=log_prob)
        f.create_dataset('true_log_prob_x', data=true_log_prob)
        f.create_dataset('samples_theta', data=samples)
        f.create_dataset('infer_theta', data=infer_theta)
    IPython.embed()


    '''
    plot HMC inference test results
    '''
    alpha_vec = np.concatenate((np.linspace(0.00, 0.994, num=100), np.linspace(0.995, 1.0, num=51)))
    coverage_gauss, coverage_gauss_lo, coverage_gauss_hi = run_inference_test(alpha_vec, log_prob, true_log_prob,
                                                                              title='Gaussian Lhood MSE', show=False)

    x_size = 3.5
    dpi_value = 200
    plt_params = {'legend.fontsize': 7,
                  'legend.frameon': False,
                  'axes.labelsize': 8,
                  'axes.titlesize': 8,
                  'figure.titlesize': 8,
                  'xtick.labelsize': 7,
                  'ytick.labelsize': 7,
                  'lines.linewidth': 1,
                  'lines.markersize': 2,
                  'errorbar.capsize': 3,
                  'font.family': 'serif',
                  # 'text.usetex': True,
                  'xtick.minor.visible': True,
                  }
    plt.rcParams.update(plt_params)
    print('plotting')

    inference_fig = plt.figure(figsize=(x_size, x_size * .9), constrained_layout=True,
                               dpi=dpi_value,
                               )
    grid = inference_fig.add_gridspec(
        nrows=1, ncols=1,
    )

    skew_ax = inference_fig.add_subplot(grid[0])

    skew_ax.plot(alpha_vec, coverage_gauss, color='black', linestyle='solid', label='inference test points',
                 zorder=10)
    skew_ax.fill_between(alpha_vec, coverage_gauss_lo, coverage_gauss_hi, facecolor='grey', alpha=0.8, zorder=3)
    x_vec = np.linspace(0.0, 1.0, 11)
    skew_ax.plot(x_vec, x_vec, linewidth=1.5, color='red', linestyle=(0, (5, 10)), zorder=20, label='inferred model')

    skew_ax.set_xlim((-0.01, 1.01))
    skew_ax.set_ylim((-0.01, 1.01))
    skew_ax.set_xlabel(r'$P_{{\rm true}}$', fontsize=16)
    skew_ax.set_ylabel(r'$P_{{\rm inf}}$', fontsize=16)

    out_path_plot = f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{z_string}/'
    inference_fig.suptitle(f'{note}')
    inference_fig.savefig(out_path_plot + f'{save_name}.png')
    print(f'plot saved as: {save_name}.png')