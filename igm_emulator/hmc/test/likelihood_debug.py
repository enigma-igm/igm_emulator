import numpy as np
import jax.numpy as jnp
import dill
import sys
import os
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator')
from emulator_run import nn_emulator
sys.path.append(os.path.expanduser('~') + '/wdm/correlation')
import matplotlib.pyplot as plt
from matplotlib import cm
import h5py
import IPython
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/hmc')
from nn_hmc_3d_x import NN_HMC_X

'''
Read in smaller bin parameters grid
'''
true_temp_idx = 11
true_gamma_idx = 4
true_fobs_idx = 7
n_inference = 5

# small bin 4 -> 3
zstr = 'z54'
skewers_per_data = 20 #17->20
n_covar = 500000
bin_label = '_set_bins_3'
in_path_molly = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final_135/{zstr}/' 
#change path from f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'

# get initial grid
in_name_h5py = f'correlation_temp_fluct_skewers_2000_R_30000_nf_9_dict{bin_label}.hdf5'
with h5py.File(in_path_molly + in_name_h5py, 'r') as f:
    params = dict(f['params'].attrs.items())
fobs = params['average_observed_flux']
R_value = params['R']
v_bins = params['v_bins']
t_0s = 10.**params['logT_0']
gammas = params['gamma']
n_temps = len(t_0s)
n_gammas = len(gammas)
n_f = len(fobs)

noise_idx = 0
like_name_0 = f'likelihood_dicts_R_30000_nf_9_T{true_temp_idx}_G{true_gamma_idx}_SNR0_F{true_fobs_idx}_ncovar_500000_P{skewers_per_data}{bin_label}.p'
like_dict_0 = dill.load(open(in_path_molly + like_name_0, 'rb'))
in_name_new_params = f'new_covariances_dict_R_30000_nf_9_ncovar_{n_covar}_' \
                     f'P{skewers_per_data}{bin_label}_params.p'
new_param_dict = dill.load(open(in_path_molly + in_name_new_params, 'rb'))
new_temps = new_param_dict['new_temps']
new_gammas = new_param_dict['new_gammas']
new_fobs = new_param_dict['new_fobs']

n_new_t = (len(new_temps) - 1)/(len(t_0s) - 1) - 1
n_new_g = (len(new_gammas) - 1)/(len(gammas) - 1) - 1
n_new_f = (len(new_fobs) - 1)/(len(fobs) - 1) - 1
new_models_np = np.empty([len(new_temps), len(new_gammas), len(new_fobs), len(v_bins)])

for old_t_below_idx in range(n_temps - 1):
    print(f'{old_t_below_idx / (n_temps - 1) * 100}%')
    for old_g_below_idx in range(n_gammas - 1):
        for old_f_below_idx in range(n_f - 1):
            fine_dict_in_name = f'new_covariances_dict_R_{int(R_value)}_nf_{n_f}_T{old_t_below_idx}_' \
                                f'G{old_g_below_idx}_SNR0_F{old_f_below_idx}_ncovar_{n_covar}_' \
                                f'P{skewers_per_data}{bin_label}.p'
            fine_dict = dill.load(open(in_path_molly + fine_dict_in_name, 'rb'))
            new_temps_small = fine_dict['new_temps']
            new_gammas_small = fine_dict['new_gammas']
            new_fobs_small = fine_dict['new_fobs']
            new_models_small = fine_dict['new_models']
            new_covariances_small = fine_dict['new_covariances']
            new_log_dets_small = fine_dict['new_log_dets']
            if old_t_below_idx == n_temps - 2:
                added_t_range = n_new_t + 2
            else:
                added_t_range = n_new_t + 1
            if old_g_below_idx == n_gammas - 2:
                added_g_range = n_new_g + 2
            else:
                added_g_range = n_new_g + 1
            if old_f_below_idx == n_f - 2:
                added_f_range = n_new_f + 2
            else:
                added_f_range = n_new_f + 1
            # print(added_f_range)
            for added_t_idx in range(int(added_t_range)):
                for added_g_idx in range(int(added_g_range)):
                    for added_f_idx in range(int(added_f_range)):
                        final_t_idx = int((old_t_below_idx * (n_new_t + 1)) + added_t_idx)
                        final_g_idx = int((old_g_below_idx * (n_new_g + 1)) + added_g_idx)
                        final_f_idx = int((old_f_below_idx * (n_new_f + 1)) + added_f_idx)
                        new_models_np[final_t_idx, final_g_idx, final_f_idx, :] = new_models_small[added_t_idx, added_g_idx, added_f_idx, :]
new_models = jnp.array(new_models_np)


print(f'fobs grid: {fobs}')

# ## Fine grid
temps_plot = np.empty([(n_temps-1) * 4 + 1])
for idx, temp in enumerate(t_0s[:-1]):
    new_idx = idx * 4
    for new_add_idx in range(4):
        temps_plot[new_idx + new_add_idx] = temp + (np.diff(t_0s)[idx] * new_add_idx / 4.)
temps_plot[-1] = t_0s[-1]

gammas_plot = np.empty([(n_gammas-1) * 4 + 1])
for idx, gamma in enumerate(gammas[:-1]):
    new_idx = idx * 4
    for new_add_idx in range(4):
        gammas_plot[new_idx + new_add_idx] = gamma + (np.diff(gammas)[idx] * new_add_idx / 4.)
gammas_plot[-1] = gammas[-1]

fobs_plot = np.empty([(n_f-1) * 4 + 1])
for idx, fob in enumerate(fobs[:-1]):
    new_idx = idx * 4
    for new_add_idx in range(4):
           fobs_plot[new_idx + new_add_idx] = fob + (np.diff(fobs)[idx] * new_add_idx / 4.)
fobs_plot[-1] = fobs[-1]

print(temps_plot.shape,gammas_plot.shape,fobs_plot.shape)

'''
Molly's model
temp, gamma, ave_f = theta
'''
def return_idx(value, all_values):
    the_min_value = jnp.min(all_values)
    the_range = jnp.max(all_values) - the_min_value
    scaled_value = (value - the_min_value) / the_range * (len(all_values) - 1)
    nearest_idx = int(jnp.round(scaled_value))
    return nearest_idx
def get_molly_model_nearest(theta,
                                  fine_temps=new_temps, fine_gammas=new_gammas, fine_fobs=new_fobs,
                                  fine_models=new_models):
    temp, gamma, ave_f = theta
    temp_idx = return_idx(temp, fine_temps)
    gamma_idx = return_idx(gamma, fine_gammas)
    fobs_idx = return_idx(ave_f, fine_fobs)
    model = fine_models[temp_idx, gamma_idx, fobs_idx, :]
    return model
def log_likelihood_molly(theta, corr, theta_covariance=like_dict_0['covariance']
                         , true_log_det=like_dict_0['log_determinant']):
    # temp, g, ave_f = theta
    model = get_molly_model_nearest(theta)
    diff = corr - model
    nbins = len(corr)
    log_like = -(jnp.dot(diff, jnp.linalg.solve(theta_covariance, diff)) + true_log_det + nbins * jnp.log(2.0 * jnp.pi)) / 2.0
    return diff.mean(), log_like

'''
Linda's model
ave_f, temp, gamma = theta_linda
'''
in_path_linda = '/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/'
emu_name = f'{zstr}_best_param_training_768_bin59.p' #small bins
best_params = dill.load(open(in_path_linda + emu_name, 'rb'))
nn_x = NN_HMC_X(v_bins, best_params, t_0s, gammas, fobs, like_dict_0)

def get_linda_model(theta, best_params_function=best_params):
    theta_linda = (theta[2], theta[0], theta[1])
    model = nn_emulator(best_params_function, theta_linda)
    return model

def log_likelihood(theta, corr):
    '''
    Args:
        x: dimensionless parameters
        flux: observed flux
    Returns:
        log_likelihood: log likelihood
    '''
    model = get_linda_model(theta) #theta is in physical dimension for this function
    #model = get_molly_model_nearest(theta) #replace same model

    new_covariance = nn_x.like_dict['covariance']
    log_determinant = nn_x.like_dict['log_determinant']

    diff = corr - model
    nbins = len(nn_x.vbins)
    log_like = -(jnp.dot(diff, jnp.linalg.solve(new_covariance, diff)) + log_determinant + nbins * jnp.log(
        2.0 * jnp.pi)) / 2.0
    #print(f'Log_likelihood={log_like}')
    return diff.mean(), log_like

def log_likelihood_linda(theta, corr):
    '''

    Parameters
    ----------
    theta: temp, gamma, ave_f
    corr

    Returns
    -------
    log_likelihood: log likelihood from nn_x parameter transformation
    '''
    theta = np.array(theta)
    theta_linda = (theta[2], theta[0], theta[1])
    theta_linda_x = nn_x.theta_to_x(theta_linda)
    return nn_x.log_likelihood(theta_linda_x, corr)

if __name__ == '__main__':
    '''
    Compare Likelihood functions line by line at sample points
    '''
    print('Covariance matrix and determinant compare:')
    print(nn_x.like_dict['covariance']==like_dict_0['covariance'],nn_x.like_dict['log_determinant']==like_dict_0['log_determinant'])

    # read in the mock data
    mock_name = f'mocks_R_{int(R_value)}_nf_{n_f}_T{true_temp_idx}_G{true_gamma_idx}_SNR{noise_idx}_F{true_fobs_idx}_P{skewers_per_data}{bin_label}.p'
    mocks = dill.load(open(in_path_molly + mock_name, 'rb'))

    # load sample theta
    sample = [t_0s[true_temp_idx],gammas[true_gamma_idx],fobs[true_fobs_idx]]
    sample_linda = [fobs[true_fobs_idx],t_0s[true_temp_idx],gammas[true_gamma_idx]]
    n_samples = 5
    for mock_idx in range(n_samples):
        print(f'Likelihood in theta vs. nn_x transformation for mock data{mock_idx}: {log_likelihood(sample,mocks[mock_idx])[1]==log_likelihood_linda(sample,mocks[mock_idx])}')
    ranind = np.random.randint(0,high=[len(temps_plot),len(gammas_plot),len(fobs_plot)],size=(20,3))
    for i in range(20):
        sample = [temps_plot[ranind[i,0]],gammas_plot[ranind[i,1]],fobs_plot[ranind[i,2]]]
        sample_linda = [fobs_plot[ranind[i,2]],temps_plot[ranind[i,0]],gammas_plot[ranind[i,1]]]
        print(f'Sample [T0, gamma, ave_f]: {sample}')
        print(f'Likelihood in theta vs. nn_x transformation for mock data{i}: {log_likelihood(sample,mocks[i])[1]==log_likelihood_linda(sample,mocks[i])}')
        print(f'NN_X parameter transformation [avg_f, T0, gamma]: {np.array(sample_linda)==nn_x.x_to_theta(nn_x.theta_to_x(sample_linda))}')
        print(f'Emulator model application:{get_linda_model(sample)==nn_emulator(best_params,sample_linda)}')


