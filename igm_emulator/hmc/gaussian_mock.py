from inference_test import *
import numpy as np

mock_corr = np.empty([n_inference, len(vbins)])
pbar = ProgressBar()
for mock_idx in pbar(range(n_inference)):
    closest_temp_idx = np.argmin(np.abs(T0s - true_theta_sampled[mock_idx, 0]))
    closest_gamma_idx = np.argmin(np.abs(gammas - true_theta_sampled[mock_idx, 1]))
    closest_fobs_idx = np.argmin(np.abs(fobs - true_theta_sampled[mock_idx, 2]))

    model_name = f'likelihood_dicts_R_30000_nf_9_T{closest_temp_idx}_G{closest_gamma_idx}_SNR0_F{closest_fobs_idx}_ncovar_{n_covar}_P{n_path}{bin_label}.p'
    model_dict = dill.load(open(in_path + model_name, 'rb'))
    mean = model_dict['mean_data']
    covariance = like_dict['covariance']
    rng = np.random.default_rng()
    mock_corr[mock_idx, :] = rng.multivariate_normal(mean, covariance)

out_path = '/mnt/quasar2/zhenyujin/igm_emulator/hmc/hmc_results/'
dill.dump(mock_corr, open(out_path + f'gaussian_mock_corr_inference{n_inference}.p', 'wb'))