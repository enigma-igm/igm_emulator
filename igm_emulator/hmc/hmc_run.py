from nn_hmc_3d_x import NN_HMC_X
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
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator')
from emulator_run import nn_emulator,small_bin_bool
print(f'Training for small bin: {small_bin_bool}')
from progressbar import ProgressBar
sys.path.append(os.path.expanduser('~') + '/dw_inference/dw_inference/inference')
sys.path.append(os.path.expanduser('~') + '/wdm/correlation/')
from mcmc_inference_new_linda_params_mult_file_3d import return_idx, get_model_covar_nearest

'''
load model and auto-corr
'''
redshift = 5.4
test_id = 15

T0_idx = 11 #0-14
g_idx = 4 #0-8
f_idx = 7 #0-8

# get the appropriate string and pathlength for chosen redshift
zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
z_idx = np.argmin(np.abs(zs - redshift))
z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
z_string = z_strings[z_idx]
in_path_hdf5 = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/best_params/'
f = h5py.File(in_path_hdf5 + f'z{redshift}_nn_savefile.hdf5', 'r')
if small_bin_bool==True:
    emu_name = f'{z_string}_best_param_training_768_bin59.p'
else:
    emu_name = f'{z_string}_best_param_training_768.p'
#IPython.embed()

best_params = dill.load(open(in_path_hdf5 + emu_name, 'rb'))
meanX = np.asarray(f['data']['meanX'])
stdX = np.asarray(f['data']['stdX'])
meanY = np.asarray(f['data']['meanY'])
stdY =  np.asarray(f['data']['stdY'])

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
gammas = params['gamma']
n_f = len(fobs)

noise_idx = 0
like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{g_idx}_SNR0_F{f_idx}_ncovar_{n_covar}_P{n_path}{bin_label}.p'
like_dict = dill.load(open(in_path + like_name, 'rb'))
mock_name = f'mocks_R_{int(R_value)}_nf_{n_f}_T{T0_idx}_G{g_idx}_SNR{noise_idx}_F{f_idx}_P{n_path}{bin_label}.p'
mocks = dill.load(open(in_path + mock_name, 'rb'))
theta_true = [fobs[f_idx], T0s[T0_idx], gammas[g_idx]]

in_path_model = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'
molly_name = f'z54_data_nearest_model_set_bins_4_steps_48000_mcmc_inference_5_one_prior_T{T0_idx}_G{g_idx}_F{f_idx}_R_30000.hdf5'
molly_model = h5py.File(in_path_model + molly_name, 'r')

mock_flux = mocks[0:5,:]
mean_flux = like_dict['mean_data']
new_covariance = like_dict['covariance']
model = nn_emulator(best_params, theta_true)
fig2, axs2 = plt.subplots(1, 1)
axs2.plot(vbins, model, label=f'Emulated' r'$<F>$='f'{theta_true[0]:.2f},'
                                                     r'$T_0$='f'{theta_true[1]:.2f},'
                                                     r'$\gamma$='f'{theta_true[2]:.2f}')
axs2.plot(vbins, mean_flux, label=f'mean', linestyle='--')
for i in mock_flux:
    axs2.plot(vbins, i, label=f'mock', linestyle='--',alpha = 0.5, color ='blue')
plt.title('Test overplot in data space')
plt.legend()
plt.show()

'''
Run HMC
'''
if __name__ == '__main__':
    nn_x = NN_HMC_X(vbins, best_params, T0s, gammas, fobs, like_dict)
    x_true = nn_x.theta_to_x(theta_true)
    key = random.PRNGKey(642)
    key, subkey = random.split(key)
    var_label = ['fobs', 'T0s', 'gammas']
    n_inference = 5
    pbar = ProgressBar()
    for mock_idx in pbar(range(n_inference)):
        note = f"jit_2000_4_test{test_id}_small_bins_retrain_compare_molly_mock{mock_idx}"
        flux = mocks[mock_idx, :]
        x_samples, theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean, \
        hmc_num_steps, hmc_tree_depth, total_time = nn_x.mcmc_one(key, x_true, flux)
        f_mcmc, t_mcmc, g_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                     zip(*np.percentile(theta_samples, [16, 50, 84], axis=0)))
        nn_x.save_HMC(z_string,note,f_idx, T0_idx,g_idx, f_mcmc, t_mcmc, g_mcmc, x_samples, theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean,
                 hmc_num_steps, hmc_tree_depth, total_time)

        y_error = np.sqrt(np.diag(new_covariance))
        molly_sample = molly_model['samples'][mock_idx, :, :]
        molly_flip = np.zeros(shape=molly_sample.shape)
        molly_flip[:, 0] = molly_sample[:, 2]
        molly_flip[:, 1] = molly_sample[:, 0]
        molly_flip[:, 2] = molly_sample[:, 1]

        t_molly, g_molly, f_molly = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                        zip(*np.percentile(molly_sample, [16, 50, 84], axis=0)))
        molly_infer, covar, log_det = get_model_covar_nearest([t_molly[0], g_molly[0], f_molly[0]])

        corner_fig = corner.corner(np.array(theta_samples), levels=(0.68, 0.95), labels=var_label,
                                   truths=np.array(theta_true), truth_color='red', show_titles=True,
                                   title_kwargs={"fontsize": 15}, label_kwargs={'fontsize': 20},
                                   data_kwargs={'ms': 1.0, 'alpha': 0.1}, hist_kwargs=dict(density=True))
        corner.corner(molly_flip, levels=(0.68, 0.95), fig=corner_fig, color='blue',hist_kwargs=dict(density=True))
        plt_params = {'legend.fontsize': 7,
                      'legend.frameon': False,
                      'axes.labelsize': 12,
                      'axes.titlesize': 12,
                      'figure.titlesize': 12,
                      'xtick.labelsize': 12,
                      'ytick.labelsize': 12,
                      'lines.linewidth': .7,
                      'lines.markersize': 2.3,
                      'lines.markeredgewidth': .9,
                      'errorbar.capsize': 2,
                      'font.family': 'serif',
                      # 'text.usetex': True,
                      'xtick.minor.visible': True,
                      }
        x_size = 5
        dpi_value = 200
        plt.rcParams.update(plt_params)

        fit_fig = plt.figure(figsize=(x_size * 2., x_size * .65), constrained_layout=True,
                             dpi=dpi_value)
        grid = fit_fig.add_gridspec(nrows=1, ncols=1)
        fit_axis = fit_fig.add_subplot(grid[0])

        inds = np.random.randint(len(theta_samples), size=100)
        for idx, ind in enumerate(inds):
            sample = theta_samples[ind]
            model_plot = nn_emulator(best_params, sample)
            molly, co, log_d = get_model_covar_nearest(molly_sample[ind])
            if idx == 0:
                fit_axis.plot(vbins, model_plot, c="b", lw=.7, alpha=0.12, zorder=1, label='Posterior Draws')
                #fit_axis.plot(vbins, molly, c="yellow", lw=.7, alpha=0.1, zorder=1, label='Old Posterior Draws')
            else:
                fit_axis.plot(vbins, model_plot, c="b", lw=.7, alpha=0.12, zorder=1)
                #fit_axis.plot(vbins, molly, c="yellow", lw=.7, alpha=0.1, zorder=1)
        max_P = max(lnP)
        max_P_idx = [index for index, item in enumerate(lnP) if item == max_P]
        print(f'max_P:{theta_samples[max_P_idx, :][0]}')
        print(f'inferred:{[f_mcmc[0], t_mcmc[0], g_mcmc[0]]}')
        inferred_model_plot = nn_emulator(best_params, [f_mcmc[0], t_mcmc[0], g_mcmc[0]])
        max_P_model = nn_emulator(best_params, theta_samples[max_P_idx, :][0])
        mean_flux = like_dict['mean_data']
        fit_axis.plot(vbins, inferred_model_plot, c="r", label='Inferred Model', zorder=5, lw=1,
                      path_effects=[pe.Stroke(linewidth=1.25, foreground='k'), pe.Normal()])
        fit_axis.plot(vbins, mean_flux, c="lightgreen", ls='--', label='True Model', zorder=2, lw=1.75,
                      path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])
        fit_axis.plot(vbins, max_P_model, c="gold", label='Max Probability Model', zorder=3, lw=1,
                      path_effects=[pe.Stroke(linewidth=1.25, foreground='k'), pe.Normal()])
        fit_axis.errorbar(vbins, flux,
                          yerr=y_error,
                          color='k', marker='.', linestyle=' ', zorder=1, capsize=0,
                          label='Mock Data')
        #fit_axis.plot(vbins, molly_infer, c="m", label='Old Model', zorder=4, lw=1, path_effects=[pe.Stroke(linewidth=1.25, foreground='k'), pe.Normal()])
        fit_axis.text(
            500, 0.0248,
            'True Model \n' + r'$\langle F \rangle$' + f' = {np.round(theta_true[0], decimals=4)}' + f'\n $T_0$ = {int(theta_true[1])} K \n $\gamma$ = {np.round(theta_true[2], decimals=3)} \n',
            {'color': 'lightgreen', 'fontsize': 10},
        )

        fit_axis.text(
            1000, 0.024,
            'Inferred Model \n' + r'$\langle F \rangle$' + f' = {np.round(f_mcmc[0], decimals=4)}$^{{+{np.round(f_mcmc[1], decimals=4)}}}_{{-{np.round(f_mcmc[2], decimals=4)}}}$' +
            f'\n $T_0$ = {int(t_mcmc[0])}$^{{+{int(t_mcmc[1])}}}_{{-{int(t_mcmc[2])}}}$ K'
            f'\n ' + r'$\gamma$' + f' = {np.round(g_mcmc[0], decimals=3)}$^{{+{np.round(g_mcmc[1], decimals=3)}}}_{{-{np.round(g_mcmc[2], decimals=3)}}}$\n',
            {'color': 'r', 'fontsize': 10},
        )

        fit_axis.text(
            1510, 0.026,
            tabulate([[r' $R_2$', #np.round(r2_score(flux, molly_infer), decimals=4),
                       np.round(r2_score(flux, max_P_model), decimals=4)],
                      ['MSE', #np.format_float_scientific(mean_squared_error(flux, molly_infer), precision=3),
                       np.format_float_scientific(mean_squared_error(flux, max_P_model), precision=3)],
                      ['Distance', #np.format_float_scientific(minkowski(flux, molly_infer), precision=3),
                       np.format_float_scientific(minkowski(flux, max_P_model), precision=3)]],
                     headers=['Matrices', 'Grid', 'Emulator'], tablefmt='orgtbl'),
            {'color': 'm', 'fontsize': 10},
        )
        fit_axis.set_xlim(vbins[0], vbins[-1])
        fit_axis.set_xlabel("Velocity (km/s)")
        fit_axis.set_ylabel("Correlation Function")
        fit_axis.legend()
        out_path = f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{z_string}/'
        fit_fig.savefig(out_path + f'model_fit_{note}.pdf')
        corner_fig.savefig(out_path + f'corner_{note}.pdf')
        print('Figures saved.')