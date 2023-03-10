import dill
import numpy as np
import h5py
from nn_hmc_3d_x import NN_HMC_X
from progressbar import ProgressBar
import os
import matplotlib.pyplot as plt
import IPython
'''
load models and grid
'''
redshift = 5.4

# get the appropriate string and pathlength for chosen redshift
zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
z_idx = np.argmin(np.abs(zs - redshift))
z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
z_string = z_strings[z_idx]
in_path_hdf5 = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/best_params/'
f = h5py.File(in_path_hdf5 + f'z{redshift}_nn_savefile.hdf5', 'r')
emu_name = f'{z_string}_best_param_training_768.p'

in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'
n_paths = np.array([17, 16, 16, 15, 15, 15, 14]) #skewers_per_data
n_path = n_paths[z_idx]
param_in_path = '/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/'
param_dict = dill.load(open(param_in_path + f'{z_string}_params.p', 'rb'))

R_value = 30000.
skewers_use = 2000
n_flux = 9
bin_label = '_set_bins_4'
added_label = ''
temp_param_dict_name = f'correlation_temp_fluct_{added_label}skewers_{skewers_use}_R_{int(R_value)}_nf_{n_flux}_dict_set_bins_4.hdf5'
with h5py.File(in_path + temp_param_dict_name, 'r') as f:
    params = dict(f['params'].attrs.items())
vbins = params['v_bins']

fobs = param_dict['fobs']  # average observed flux <F> ~ Gamma_HI
log_T0s = param_dict['log_T0s']  # log(T_0) from temperature - density relation
T0s = np.power(10,log_T0s)
gammas = param_dict['gammas']  # gamma from temperature - density relation

best_params = dill.load(open(in_path_hdf5 + emu_name, 'rb'))
T0_idx = 11 #0-14
g_idx = 4 #0-8
f_idx = 7 #0-8


one_cov_name = "z54_data_nearest_model_set_bins_4_log_like_on_grid_5_one_prior_T11_G4_F7_R_30000_one_covariance.hdf5"
one_cov_dict = h5py.File(in_path + one_cov_name, 'r') #['fobs_grid', 'gammas_grid', 'log_likelihood_grid', 'temps_grid', 'true_theta']
like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{g_idx}_SNR0_F{f_idx}_ncovar_500000_P{n_path}_set_bins_4.p'
like_dict = dill.load(open(in_path + like_name, 'rb')) #['mean_data', 'covariance', 'correlation', 'condition_number', 'inv_covariance', 'sign', 'log_determinant']

mock_name = f'mocks_R_30000_nf_9_T{T0_idx}_G{g_idx}_SNR0_F{f_idx}_P{n_path}_set_bins_4.p'
mocks = dill.load(open(in_path + mock_name, 'rb'))
theta_true = [fobs[f_idx], T0s[T0_idx], gammas[g_idx]]

mock_flux = mocks[0:5,:]
mean_flux = like_dict['mean_data']

'''
Load molly likelihood grid
'''
fobs_grid = one_cov_dict['fobs_grid']
gammas_grid = one_cov_dict['gammas_grid']
temps_grid = one_cov_dict['temps_grid']

molly_loglike_grid = one_cov_dict['log_likelihood_grid']

'''
Compute likelihood in the same grid
'''
nn_x = NN_HMC_X(vbins,best_params,T0s,gammas,fobs,like_dict)
x_true = nn_x.theta_to_x(theta_true)
n_inference = 5
linda_loglike_grid = np.zeros([n_inference, len(fobs_grid), len(temps_grid), len(gammas_grid)])
pbar = ProgressBar()
print("START RUNNING")

g_plot_idx = int(np.floor(len(gammas_grid)/2.))
f_plot_idx = int(np.floor(len(fobs_grid)/2.))
for mock_idx in pbar(range(n_inference)):
    flux = mocks[mock_idx, :]
    for t_plot_idx, t_plot in enumerate(temps_grid):
        linda_loglike_grid[mock_idx, f_plot_idx, t_plot_idx, g_plot_idx] = nn_x.log_likelihood((fobs_grid[f_plot_idx], t_plot, gammas_grid[g_plot_idx]),
                                                                                               flux)
'''
for mock_idx in pbar(range(n_inference)):
    flux = mocks[mock_idx, :]
    for f_plot_idx, f_plot in enumerate(fobs_grid):
        for t_plot_idx, t_plot in enumerate(temps_grid):
                for g_plot_idx, g_plot in enumerate(gammas_grid):
                        linda_loglike_grid[mock_idx, f_plot_idx, t_plot_idx, g_plot_idx] =  nn_x.log_likelihood((f_plot, t_plot, g_plot), flux)
'''
print('DONE')
print(linda_loglike_grid.shape)
'''
Plotting the likelihood grid in temperature
'''
'''
x_size = 3.5
dpi_value = 200
plt_params = {'legend.fontsize': 7,
              'legend.frameon': False,
              'axes.labelsize': 8,
              'axes.titlesize': 6.5,
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

# plot one 1d slice - temps only
g_plot_idx = int(np.floor(len(gammas_grid)/2.))
f_plot_idx = int(np.floor(len(fobs_grid)/2.))

slice_fig = plt.figure(figsize=(x_size, x_size*.77*5.*.5), constrained_layout=True,
                                dpi=dpi_value,
                                )
grid = slice_fig.add_gridspec(
    nrows=5, ncols=1, # width_ratios=[20, 20, 20, 20, 20, 1],
)

for mock_idx in range(n_inference):
    axes = slice_fig.add_subplot(grid[mock_idx])
    axes.plot(temps_grid, linda_loglike_grid[mock_idx, f_plot_idx, :, g_plot_idx])

    axes.set_ylabel('log(likelihood)')
    axes.set_title(f'mock {mock_idx}')

axes.set_xlabel('$T_0$ (K)')
axes.show()

out_path = os.path.expanduser('~') + '/igm_emulator/igm_emulator/hmc/test/'
save_name = f'temperature_log_like_linda'
slice_fig.savefig(out_path + f'{save_name}.pdf')
'''
out_path = os.path.expanduser('~') + '/igm_emulator/igm_emulator/hmc/test/'
dill.save(linda_loglike_grid, open(out_path + f'linda_loglike_grid_{emu_name}.p', 'wb'))
print('PLOT AND LIKELIHOOD GRID SAVED')


