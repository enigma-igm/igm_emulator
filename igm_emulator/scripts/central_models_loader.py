import h5py
import os
import numpy as np
import glob


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
n_inference = 4000
n_plot_rows = 2

# Compare to Molly's mocks
seed = 203
rand = np.random.RandomState(seed)  # if seed is None else seed
mock_indices = rand.choice(np.arange(n_inference), replace=False, size=len(redshifts) * n_plot_rows)

in_path_out = f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/hmc_results/central_models/'
in_path_read = os.path.expanduser('~') + f'/igm_emulator/igm_emulator/hmc/hmc_results/'

samples_temp = np.empty([n_plot_rows, len(redshifts), n_inference])
samples_gamma = np.empty([n_plot_rows, len(redshifts), n_inference])

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
        name = f'{zstr}_F{f_idx}_T0{T0_idx}_G{g_idx}_central_mock_{mock_idx}.hdf5'
        with h5py.File(in_path_read + name, 'r') as f:
            # ["<F>", "$T_0$", "$\gamma$"]
            samples_temp[i, redshift_idx,:] = f['theta_samples'][:, 1]
            samples_gamma[i, redshift_idx, :] = f['theta_samples'][:, 2]

'''
Save what we have
'''

out_file_tag = f'hmc_inference_{int(n_inference)}'
in_name_inference = f'{z_strings[0]}_{z_strings[-1]}_F{f_idx}_T0{T0_idx}_G{g_idx}_central_model_{out_file_tag}.hdf5'

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
print(f'saved central models for all redshift at {in_path_out}')



