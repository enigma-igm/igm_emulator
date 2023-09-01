from inference_test import var_tag, T0s, gammas, fobs, n_inference, vbins, in_path, like_dict, best_params, n_params,n_covar, n_path, bin_label
import numpy as np
import dill
import jax.random as random
from progressbar import ProgressBar
import igm_emulator as emu

# get n_inference sampled parameters
true_theta_sampled = np.empty([n_inference, n_params])
rng = np.random.default_rng(36)
seed = rng.integers(0, 100,3)

true_temp_x = random.uniform(random.PRNGKey(seed[0]),(n_inference,), minval=T0s[0], maxval=T0s[-1]+0.1)

true_gamma_x = random.uniform(random.PRNGKey(seed[1]),(n_inference,), minval=gammas[0], maxval=gammas[-1]+0.1)

true_fobs_x = random.uniform(random.PRNGKey(seed[2]),(n_inference,), minval=fobs[0], maxval=fobs[-1]+0.1)

true_theta_sampled[:, 0] =true_temp_x #nn_x.x_to_theta(true_temp_x)
true_theta_sampled[:, 1] =true_gamma_x #nn_x.x_to_theta(true_gamma_x)
true_theta_sampled[:, 2] =true_fobs_x #nn_x.x_to_theta(true_fobs_x)

#get n_inference mock correlation functions
mock_corr = np.empty([n_inference, len(vbins)])
true_theta = np.empty([n_inference, n_params])
pbar = ProgressBar()
for mock_idx in pbar(range(n_inference)):

    closest_temp_idx = np.argmin(np.abs(T0s - true_theta_sampled[mock_idx, 0]))
    closest_gamma_idx = np.argmin(np.abs(gammas - true_theta_sampled[mock_idx, 1]))
    closest_fobs_idx = np.argmin(np.abs(fobs - true_theta_sampled[mock_idx, 2]))

    true_theta[mock_idx, :] = [fobs[closest_fobs_idx], T0s[closest_temp_idx], gammas[closest_gamma_idx]]

    #model_name = f'likelihood_dicts_R_30000_nf_9_T{closest_temp_idx}_G{closest_gamma_idx}_SNR0_F{closest_fobs_idx}_ncovar_{n_covar}_P{n_path}{bin_label}.p'
    #model_dict = dill.load(open(in_path + model_name, 'rb'))
    #mean = model_dict['mean_data'] #around emulator mean

    #true_theta[mock_idx, :] = [true_theta_sampled[mock_idx, 2], true_theta_sampled[mock_idx, 0], true_theta_sampled[mock_idx, 1]]

    mean = emu.nn_emulator(best_params, true_theta[mock_idx, :])
    covariance = like_dict['covariance']
    rng = np.random.default_rng()
    mock_corr[mock_idx, :] = rng.multivariate_normal(mean, covariance)

#save get n_inference sampled parameters and mock correlation functions
out_path = '/mnt/quasar2/zhenyujin/igm_emulator/hmc/hmc_results/'
note = 'gaussian_emulator_prior_x'

dill.dump(mock_corr, open(out_path + f'{note}_corr_inference{n_inference}_{var_tag}.p', 'wb'))
dill.dump(true_theta, open(out_path + f'{note}_theta_inference{n_inference}_{var_tag}.p', 'wb'))