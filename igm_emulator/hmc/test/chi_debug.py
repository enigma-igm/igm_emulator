import numpy as np
import jax.numpy as jnp
import dill
import sys
import os
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator')
from emulator_run import nn_emulator
sys.path.append(os.path.expanduser('~') + '/wdm/correlation')
import matplotlib.pyplot as plt
import h5py
import IPython
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/hmc')
from nn_hmc_3d_x import NN_HMC_X
from matplotlib import cm

# ## Read in smaller bin parameters and mock data
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


# ## Molly's model
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


# ## Linda's model
# get Linda's model 
#temp, gamma, ave_f = theta

sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/hmc')
from nn_hmc_3d_x import NN_HMC_X

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
    theta = np.array(theta)
    theta_linda = (theta[2], theta[0], theta[1])
    theta_linda_x = nn_x.theta_to_x(theta_linda)
    return nn_x.log_likelihood(theta_linda_x, corr)


# read in the mock data
mock_name = f'mocks_R_{int(R_value)}_nf_{n_f}_T{true_temp_idx}_G{true_gamma_idx}_SNR{noise_idx}_F{true_fobs_idx}_P{skewers_per_data}{bin_label}.p'
mocks = dill.load(open(in_path_molly + mock_name, 'rb'))

sample = [t_0s[true_temp_idx],gammas[true_gamma_idx],fobs[true_fobs_idx]]
n_samples = 5
for mock_idx in range(2):
    print(log_likelihood(sample,mocks[mock_idx]))
    print(log_likelihood_linda(sample,mocks[mock_idx]))


# ## Plot chi for two models

emu_path = os.path.expanduser('~') + f'/igm_emulator/igm_emulator/emulator/best_params/'
#change retrain param
with h5py.File(emu_path+'z5.4_nn_bin59_savefile.hdf5', 'r') as f:
    # IPython.embed()
    residual = np.asarray(f['performance']['residuals'])
    meanY = np.asarray(f['data']['meanY'])
    stdY = np.asarray(f['data']['stdY'])
    print(f['data'].keys())
    print(f['performance'].attrs.keys())

dir_lhs = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/GRID/'

test_num = '_test_89_bin59'
Y_test = dill.load(open(dir_lhs + f'{zstr}_model{test_num}.p', 'rb'))
X_test = dill.load(open(dir_lhs + f'{zstr}_param{test_num}.p', 'rb'))
#Y_test = (Y_test - meanY) / stdY

diff = residual * Y_test
chi2 = 0
chi = []
rel_err = []
chi2_molly = 0
chi_molly = []
rel_err_molly = []
print(diff.shape)
for d_i in range(Y_test.shape[0]):
    d = Y_test[d_i,:] - get_linda_model([X_test[d_i,1],X_test[d_i,2],X_test[d_i,0]])
    chi2=+jnp.dot(d, jnp.linalg.solve(like_dict_0['covariance'], d))
    #chi.append(jnp.linalg.solve(jnp.sqrt(like_dict_0['covariance']), d))
    chi.append(np.multiply(np.diagonal(like_dict_0['covariance']), d))
    rel_err.append(d/Y_test[d_i,:]*100)
    
    diff_molly = Y_test[d_i,:] - get_molly_model_nearest([X_test[d_i,1],X_test[d_i,2],X_test[d_i,0]])
    chi2_molly=+(jnp.dot(diff_molly, jnp.linalg.solve(like_dict_0['covariance'], diff_molly)))
    #chi_molly.append(jnp.linalg.solve(jnp.sqrt(like_dict_0['covariance']), diff_molly))
    chi_molly.append(np.multiply(np.diagonal(like_dict_0['covariance']), diff_molly))
    rel_err_molly.append(diff_molly/Y_test[d_i,:]*100)
    #print(diff_molly/Y_test[d_i,:]<=0)
chi = np.array(chi).T
chi_molly = np.array(chi_molly).T
rel_err = np.array(rel_err).T
rel_err_molly = np.array(rel_err_molly).T

chi2_dof = chi2/Y_test.shape[1]
chi2_molly_dof = chi2_molly/Y_test.shape[1]
print(f'chi2 square emulator: {chi2_dof},chi2 square molly: {chi2_molly_dof}')
plt.plot(v_bins,Y_test.T)

plt.show()


# In[10]:


with h5py.File(emu_path+'z5.4_nn_bin59_savefile.hdf5', 'r') as f:
    print('smaller bin')
    print(f['performance'].attrs['residuals_results'])
    print(f['performance'].attrs['R2'])

with h5py.File(emu_path+'z5.4_nn_savefile.hdf5', 'r') as f:
    print('larger bin')
    print(f['performance'].attrs['residuals_results'])
    print(f['performance'].attrs['R2'])



x_size = 4
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

plt.figure(figsize=(x_size, x_size*4), constrained_layout=True,
                                dpi=dpi_value)

fig1 = plt.subplot(3,1,1)
fig1.plot(v_bins, chi, linewidth=0.5, color = 'b'#, alpha=0.2
         ,label='Linda'
        )
fig1.plot(v_bins, chi_molly, linewidth=0.5, color = 'r', alpha=0.2
         ,label='Molly'
        )
fig1.set_xlabel(r'Velocity [$km s^{-1}$]')
fig1.set_ylabel(r'$\chi$')
#plt.title(f'%Residual plot:mean: {np.mean(diff)}; std: {np.std(diff)}')

fig2 = plt.subplot(3,1,2)
fig2.plot(v_bins, rel_err, linewidth=0.5, color = 'b'#, alpha=0.2
         #,label='linda'
        )
fig2.plot(v_bins, rel_err_molly[:,32], linewidth=0.5, color = 'r'#, alpha=0.2
         #,label='molly'
        )
#fig2.plot(v_bins, residual.T, linewidth=0.5, color = 'g', alpha=0.2
         #,label='molly'
        #)
fig2.set_xlabel(r'Velocity [$km s^{-1}$]')
fig2.set_ylabel(r'Relative error (%)')


# In[ ]:


print(X_test[:10,:])

plt.plot(v_bins,get_molly_model_nearest([X_test[:,1],X_test[:,2],X_test[:,0]]),label='molly')
plt.plot(v_bins,Y_test.T,label='test')

plt.legend()

colormap = cm.Reds
n = 3
percentiles = [68,95,99]
rel_err_perc= np.zeros((59,n))
rel_err_molly_perc = np.zeros((59,n))

for i in range(n):
    rel_err_perc[:,i]=np.percentile(rel_err,percentiles[i],axis=1)
    rel_err_molly_perc[:,i]=np.percentile(rel_err_molly,percentiles[i],axis=1)

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8,8))
for i in range(n):
    ax[0].fill_between(v_bins, rel_err_perc[:,i],color=colormap(i/n),zorder=-i,label=f'{percentiles[i]}%')
    ax[1].fill_between(v_bins, rel_err_molly_perc[:,i],color=colormap(i/n),zorder=-i)
ax[0].set_title("Percentile plot", fontsize=15)
ax[0].tick_params(labelsize=11.5)
ax[1].set_xlabel(r'Velocity [$km s^{-1}$]', fontsize=14)
ax[0].set_ylabel(r'Relative error Emulator(%)', fontsize=10)
ax[1].set_ylabel(r'Relative error Molly(%)', fontsize=10)
fig.tight_layout()
fig.legend()


# ## Print evidence of HMC
test_id = 15
molly_name = f'z54_data_nearest_model_set_bins_4_steps_48000_mcmc_inference_5_one_prior_T{true_temp_idx}_G{true_gamma_idx}_F{true_fobs_idx}_R_30000.hdf5'
molly_hmc = h5py.File(f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{zstr}/final_135/' + molly_name, 'r')
for mock_idx in range(n_inference): 
    molly_sample = molly_hmc['samples'][mock_idx,:,:]
    molly_flip = np.zeros(shape = molly_sample.shape)
    molly_flip[:,0] = molly_sample[:,2]
    molly_flip[:,1] = molly_sample[:,0]
    molly_flip[:,2] = molly_sample[:,1]
    t_molly, g_molly, f_molly = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                     zip(*np.percentile(molly_sample, [16, 50, 84], axis=0)))
    molly_infer = get_molly_model_nearest([t_molly[0], g_molly[0],f_molly[0]])
    
    linda_name = f"jit_2000_4_test{test_id}_small_bins_compare_molly_mock{mock_idx}"
    linda_hmc = h5py.File(os.path.expanduser('~') + f'/igm_emulator/igm_emulator/hmc/hmc_results/' +f'{zstr}_F{true_fobs_idx}_T0{true_temp_idx}_G{true_gamma_idx}_{linda_name}_hmc_results.hdf5', 'r')
    
    molly_evidence = np.sum(molly_hmc['log_prob'][mock_idx,:])
    linda_evidence = np.sum(linda_hmc['lnP'])
    print(f'molly_evidence: {molly_evidence}, linda_evidence: {linda_evidence}')
                      

