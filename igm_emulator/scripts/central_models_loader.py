import h5py
import os
import numpy as np
import glob
import dill


'''
GOAL:
 samples_temp[mock_idx,redshift_idx, :]
(samples_temp = f['samples'][mock_idx, n_walk, 0])
  samples_gamma[mock_idx,redshift_idx, :]
    pdf_hists_temp[mock_idx, redshift_idx, :] = hist_temp
'''


redshifts = [5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]
z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
T0_idx = 7
g_idx = 4
f_idx = 4
n_inference = 4000*4
n_plot_rows = 2
n_inference_w = 300
n_mcmc = 3500
n_walkers = 16
n_skip = 500
R_value = 30000
bin_label = '_set_bins_3'

# Compare to Molly's mocks
seed = 203
rand = np.random.RandomState(seed)  # if seed is None else seed
mock_indices = rand.choice(np.arange(100), replace=False, size=len(redshifts) * n_plot_rows)

in_path_out = f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/hmc_results/central_models/'
in_path_read = os.path.expanduser('~') + f'/igm_emulator/igm_emulator/hmc/hmc_results/'

samples_temp = np.empty([n_plot_rows, len(redshifts), n_inference])
samples_gamma = np.empty([n_plot_rows, len(redshifts), n_inference])

samples_temp_mean = np.empty([1, len(redshifts), n_inference])
samples_gamma_mean = np.empty([1, len(redshifts), n_inference])
samples_f_mean = np.empty([1, len(redshifts), n_inference])

samples_temp_mean_molly = np.empty([1, len(redshifts), n_walkers * (n_mcmc - n_skip)])
samples_gamma_mean_molly = np.empty([1, len(redshifts), n_walkers * (n_mcmc - n_skip)])
importance_weights_chain_molly = np.empty([1, len(redshifts), n_walkers * (n_mcmc - n_skip)])

for redshift_idx in range(len(redshifts)):
    print(f'z = {redshifts[redshift_idx]}')
    zstr = z_strings[redshift_idx]

    '''
    Load the central models -- random 5 mocke
    '''
    #partial_tag = f'{zstr}_F{f_idx}_T0{T0_idx}_G{g_idx}_central_mock_'
    #
    #for mock_idx, mock_result in enumerate(glob.glob(in_path_read + partial_tag + '*.hdf5')[:n_plot_rows]):
    #    with h5py.File(mock_result, 'r') as f:
    #        # ["<F>", "$T_0$", "$\gamma$"]
    #        samples_temp[mock_idx, redshift_idx,:] = f['theta_samples'][:, 1]
    #        samples_gamma[mock_idx, redshift_idx, :] = f['theta_samples'][:, 2]

    '''
    Load Molly's mocks with seed 203
    '''
    idx = np.sort(mock_indices[redshift_idx * n_plot_rows:(redshift_idx + 1) * n_plot_rows])
    for i, mock_idx in enumerate(idx):
        name = f'{zstr}_F{f_idx}_T0{T0_idx}_G{g_idx}_central_mock_{mock_idx}_Molly_hmc_results.hdf5'
        with h5py.File(in_path_read + name, 'r') as f:
            # ["<F>", "$T_0$", "$\gamma$"]
            samples_temp[i, redshift_idx,:] = f['theta_samples'][:, 1]
            samples_gamma[i, redshift_idx, :] = f['theta_samples'][:, 2]

    '''
    Load Mean models -- Emulator
    '''
    name = f'{zstr}_F{f_idx}_T0{T0_idx}_G{g_idx}_central_mean_model_hmc_results.hdf5'
    with h5py.File(in_path_read + name, 'r') as f:
        samples_temp_mean[0,redshift_idx,:] = f['theta_samples'][:, 1]
        samples_gamma_mean[0,redshift_idx, :] = f['theta_samples'][:, 2]
        samples_f_mean[0,redshift_idx, :] = f['theta_samples'][:, 0]

    '''
    Load Mean models -- Molly
    '''
    in_path_start = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final_135/{zstr}/'
    out_file_tag_model = f'steps_{int(n_walkers * (n_mcmc - n_skip))}_R_{int(R_value)}'
    in_name_model = f'{zstr}_model_T{T0_idx}_G{g_idx}_F{f_idx}_data_nearest_model_{out_file_tag_model}.hdf5'
    with h5py.File(in_path_start + in_name_model, 'r') as f:
        # ["$T_0$", "$\gamma$", "<F>"]
        samples_temp_mean_molly[0,redshift_idx,:] = f['samples'][:, 0]
        samples_gamma_mean_molly[0,redshift_idx, :] = f['samples'][:, 1]
        log_prob_model = f['log_prob'][:]
    prior_tag_w = 'sample_prior_leq'
    out_file_tag_w = f'steps_{int(n_walkers * (n_mcmc - n_skip))}_mcmc_inference_{int(n_inference_w)}_{prior_tag_w}'
    weight_name = f'{zstr}_data_nearest_model{bin_label}_{out_file_tag_w}_R_{int(R_value)}_weights.p'
    weight_dict = dill.load(open(in_path_start + weight_name, 'rb'))
    importance_weights = weight_dict['importance_weights']
    isort_model = np.argsort(log_prob_model[:])
    importance_weights_chain_molly[0, redshift_idx, isort_model] = importance_weights

'''
Save what we have
'''

#out_file_tag = f'hmc_inference_{int(n_inference)}'
out_file_tag = f'hmc_inference_{int(n_inference)}_Molly'
in_name_inference = f'{z_strings[0]}_{z_strings[-1]}_F{f_idx}_T0{T0_idx}_G{g_idx}_central_model_{out_file_tag}.hdf5'
mean_model_name = f'{z_strings[0]}_{z_strings[-1]}_F{f_idx}_T0{T0_idx}_G{g_idx}_central_model_mean.hdf5'
mean_model_molly_name = f'{z_strings[0]}_{z_strings[-1]}_F{f_idx}_T0{T0_idx}_G{g_idx}_central_model_mean_reweight_molly.hdf5'

if os.path.exists(in_path_out + in_name_inference):
    os.remove(in_path_out + in_name_inference)
    print(f'rewrite {in_path_out + in_name_inference}')
with h5py.File(in_path_out + in_name_inference, 'w') as f:
    f.create_dataset('samples_temp', data=samples_temp)
    f.create_dataset('samples_gamma', data=samples_gamma)
    f.attrs['z_strings'] = z_strings
    f.attrs['T0_idx'] = T0_idx
    f.attrs['g_idx'] = g_idx
    f.attrs['f_idx'] = f_idx
    f.attrs['n_inference'] = n_inference
    f.attrs['n_plot_rows'] = n_plot_rows
    f.close()

if os.path.exists(in_path_out + mean_model_name):
    os.remove(in_path_out + mean_model_name)
    print(f'rewrite {in_path_out + mean_model_name}')
with h5py.File(in_path_out + mean_model_name, 'w') as f:
    f.create_dataset('samples_temp', data=samples_temp_mean)
    f.create_dataset('samples_gamma', data=samples_gamma_mean)
    f.create_dataset('samples_f', data=samples_f_mean)
    f.attrs['z_strings'] = z_strings
    f.attrs['T0_idx'] = T0_idx
    f.attrs['g_idx'] = g_idx
    f.attrs['f_idx'] = f_idx
    f.attrs['n_inference'] = n_inference
    f.attrs['n_plot_rows'] = 1
    f.close()

if os.path.exists(in_path_out + mean_model_molly_name):
    os.remove(in_path_out + mean_model_molly_name)
    print(f'rewrite {in_path_out + mean_model_molly_name}')
with h5py.File(in_path_out + mean_model_molly_name, 'w') as f:
    f.create_dataset('samples_temp', data=samples_temp_mean_molly)
    f.create_dataset('samples_gamma', data=samples_gamma_mean_molly)
    f.create_dataset('importance_weights', data=importance_weights_chain_molly)
    f.attrs['z_strings'] = z_strings
    f.attrs['T0_idx'] = T0_idx
    f.attrs['g_idx'] = g_idx
    f.attrs['f_idx'] = f_idx
    f.attrs['n_inference'] = int(n_walkers * (n_mcmc - n_skip))
    f.attrs['n_plot_rows'] = 1
    f.close()

print(f'saved central models for all redshift at {in_path_out}')



