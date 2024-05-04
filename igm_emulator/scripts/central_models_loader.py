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

in_path_out = f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/hmc_results/central_models/'
in_path_read = os.path.expanduser('~') + f'/igm_emulator/igm_emulator/hmc/hmc_results/'

for redshift_idx in range(len(redshifts)):
    print(f'z = {redshifts[redshift_idx]}')
    zstr = z_strings[redshift_idx]

    partial_tag = f'{zstr}_F{f_idx}_T0{T0_idx}_G{g_idx}_central_mock_'

    for mock_result in glob.glob(in_path_read + partial_tag + '*.hdf5'):
        print(mock_result)

    '''
    run_tag = f'data_nearest_model{bin_label}'

    # out_file_tag = f'walkers_{int(n_walkers)}_mcmc_inference_{int(n_inference)}_{prior_tag}_R_{int(R_value)}'
    out_file_tag = f'steps_{int(n_walkers * (n_mcmc - n_skip))}_mcmc_inference_{int(n_inference)}_{prior_tag}_R_{int(R_value)}'

    in_name_inference = f'{z_strings[redshift_idx]}_{run_tag}_{out_file_tag}.hdf5'

    with h5py.File(in_path_out + f'{z_strings[redshift_idx]}/' + in_name_inference, 'w') as f:
        # true_theta = f['true_theta'][:, :]
        log_prob = f['log_prob'][np.sort(mock_indices[redshift_idx * n_plot_rows:(redshift_idx + 1) * n_plot_rows]),
                   :]  # n_inference, n_total_steps
        # true_log_prob = f['true_log_prob'][:]
        samples_temp = f['samples'][np.sort(mock_indices[redshift_idx * n_plot_rows:(redshift_idx + 1) * n_plot_rows]),
                       :, 0]
        samples_gamma = f['samples'][np.sort(mock_indices[redshift_idx * n_plot_rows:(redshift_idx + 1) * n_plot_rows]),
                        :, 1]
                        
    '''