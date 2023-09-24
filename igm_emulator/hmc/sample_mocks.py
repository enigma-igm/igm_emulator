from inference_test import var_tag, T0s, gammas, fobs, n_inference, vbins,R_value, n_f, noise_idx, in_path, like_dict, best_params, n_params,n_covar, n_path, bin_label
import numpy as np
import dill
import jax.random as random
from progressbar import ProgressBar
import igm_emulator as emu

emu_test = False #True: inference test on emulator; False: inference test on mocks
ngp = True #True: nearest grid point mocks/emulator; False: emulator
gaussian = True #True: gaussianized mocks/emulator; False: forward-modeled mocks

if gaussian == False:
    ngp = True
    note = 'forward_mocks_ngp_prior_diff_covar'
else:
    if ngp == True and emu_test == False:
        note = 'gaussian_ngp_mocks_prior_diff_covar'
    elif ngp == True and emu_test == True:
        note = 'gaussian_ngp_emulator_prior_diff_covar'
    else:
        note = 'gaussian_emulator_prior_diff_covar'

print('Sampling parameters from priors')

# get n_inference sampled parameters
true_theta_sampled = np.empty([n_inference, n_params])
rng = random.PRNGKey(36)

rng, init_rng = random.split(rng)
true_temp = random.uniform(init_rng,(n_inference,), minval=T0s[0], maxval=T0s[-1])

rng, init_rng = random.split(rng)
true_gamma = random.uniform(init_rng,(n_inference,), minval=gammas[0], maxval=gammas[-1])

rng, init_rng = random.split(rng)
true_fob = random.uniform(init_rng,(n_inference,), minval=fobs[0], maxval=fobs[-1])

true_theta_sampled[:, 1] =true_temp
true_theta_sampled[:, 2] =true_gamma
true_theta_sampled[:, 0] =true_fob

#get n_inference mock correlation functions
mock_corr = np.empty([n_inference, len(vbins)])
mock_covar = np.empty([n_inference, len(vbins),len(vbins)])
true_theta = np.empty([n_inference, n_params])
pbar = ProgressBar()
if gaussian:
    for mock_idx in pbar(range(n_inference)):

        closest_temp_idx = np.argmin(np.abs(T0s - true_theta_sampled[mock_idx, 1]))
        closest_gamma_idx = np.argmin(np.abs(gammas - true_theta_sampled[mock_idx, 2]))
        closest_fobs_idx = np.argmin(np.abs(fobs - true_theta_sampled[mock_idx, 0]))
        if ngp:
            true_theta[mock_idx, :] = [fobs[closest_fobs_idx], T0s[closest_temp_idx], gammas[closest_gamma_idx]]
            model_name = f'likelihood_dicts_R_30000_nf_9_T{closest_temp_idx}_G{closest_gamma_idx}_SNR0_F{closest_fobs_idx}_ncovar_{n_covar}_P{n_path}{bin_label}.p'
            model_dict = dill.load(open(in_path + model_name, 'rb'))
            mean = model_dict['mean_data']
            cov = model_dict['covariance']
            if emu_test:
                mean = emu.nn_emulator(best_params, true_theta[mock_idx, :])
                cov = like_dict['covariance']
        elif emu_test:
            true_theta[mock_idx, :] = true_theta_sampled[mock_idx, :] #for emulator, true_theta = true_theta_sampled noth off-grid
            mean = emu.nn_emulator(best_params, true_theta[mock_idx, :])
            cov = like_dict['covariance']
        else:
            raise Exception("For off-grid inference test, must use emulator.")
        rng = random.PRNGKey(42)
        mock_corr[mock_idx, :] = random.multivariate_normal(rng, mean, cov)
        mock_covar[mock_idx, :, :] = cov

else:
    for mock_idx in pbar(range(n_inference)):
        closest_temp_idx = np.argmin(np.abs(T0s - true_theta_sampled[mock_idx, 1]))
        closest_gamma_idx = np.argmin(np.abs(gammas - true_theta_sampled[mock_idx, 2]))
        closest_fobs_idx = np.argmin(np.abs(fobs - true_theta_sampled[mock_idx, 0]))

        true_theta[mock_idx, :] = [fobs[closest_fobs_idx], T0s[closest_temp_idx], gammas[closest_gamma_idx]]
    mock_name = f'mocks_R_{int(R_value)}_nf_{n_f}_T{closest_temp_idx}_G{closest_gamma_idx}_SNR{noise_idx}_F{closest_fobs_idx}_P{n_path}{bin_label}.p'
    mocks = dill.load(open(in_path + mock_name, 'rb'))
    model_name = f'likelihood_dicts_R_30000_nf_9_T{closest_temp_idx}_G{closest_gamma_idx}_SNR0_F{closest_fobs_idx}_ncovar_{n_covar}_P{n_path}{bin_label}.p'
    model_dict = dill.load(open(in_path + model_name, 'rb'))
    cov = model_dict['covariance']
    mock_corr[mock_idx, :] = mocks[mock_idx, :]
    mock_covar[mock_idx, :, :] = cov

#save get n_inference sampled parameters and mock correlation functions
out_path = '/mnt/quasar2/zhenyujin/igm_emulator/hmc/hmc_results/'

dill.dump(mock_corr, open(out_path + f'{note}_corr_inference{n_inference}_{var_tag}.p', 'wb'))
dill.dump(true_theta, open(out_path + f'{note}_theta_inference{n_inference}_{var_tag}.p', 'wb'))
dill.dump(true_theta_sampled, open(out_path + f'{note}_theta_sampled_inference{n_inference}_{var_tag}.p', 'wb'))
dill.dump(mock_covar, open(out_path + f'{note}_covar_inference{n_inference}_{var_tag}.p', 'wb'))