from likelihood_debug import log_likelihood_molly, get_molly_model_nearest, get_linda_model,v_bins,zstr,like_dict_0
import sys
import os
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator')
from emulator_run import nn_emulator
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/hmc')
from nn_hmc_3d_x import NN_HMC_X
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import cm
import h5py
import dill

'''
Load testing dataset
'''
emu_path = os.path.expanduser('~') + f'/igm_emulator/igm_emulator/emulator/best_params/'
#change retrain param
with h5py.File(emu_path+'z5.4_nn_bin59_savefile.hdf5', 'r') as f:
    residual = np.asarray(f['performance']['residuals'])
    meanY = np.asarray(f['data']['meanY'])
    stdY = np.asarray(f['data']['stdY'])
    print(f['data'].keys())
    print(f['performance'].attrs.keys())

dir_lhs = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/GRID/'
out_path = os.path.expanduser('~') + '/igm_emulator/igm_emulator/hmc/plots/'

test_num = '_test_89_bin59'
train_num = '_training_768_bin59'
Y_test = dill.load(open(dir_lhs + f'{zstr}_model{test_num}.p', 'rb'))
X_test = dill.load(open(dir_lhs + f'{zstr}_param{test_num}.p', 'rb'))
X_train = dill.load(open(dir_lhs + f'{zstr}_param{train_num}.p', 'rb'))

'''
Comput chi2 and chi
'''
chi2 = 0
chi = []
rel_err = []  # in percentage
model_linda = []
rel_precision = []
chi2_molly = 0
chi_molly = []
rel_err_molly = []  # in percentage
model_molly = []
count = 0
for d_i in range(Y_test.shape[0]):
    d = Y_test[d_i, :] - get_linda_model([X_test[d_i, 1], X_test[d_i, 2], X_test[d_i, 0]])
    model_linda.append(get_linda_model([X_test[d_i, 1], X_test[d_i, 2], X_test[d_i, 0]]))
    chi2 = +jnp.dot(d, jnp.linalg.solve(like_dict_0['covariance'], d))
    # chi.append(jnp.linalg.solve(jnp.sqrt(like_dict_0['covariance']), d))
    chi.append(d/np.sqrt(np.diagonal(like_dict_0['covariance'])))
    rel_precision.append(Y_test[d_i, :]/np.sqrt(np.diagonal(like_dict_0['covariance'])))
    rel_err.append(d / Y_test[d_i, :] * 100)

    diff_molly = Y_test[d_i, :] - get_molly_model_nearest([X_test[d_i, 1], X_test[d_i, 2], X_test[d_i, 0]])
    model_molly.append(get_molly_model_nearest([X_test[d_i, 1], X_test[d_i, 2], X_test[d_i, 0]]))
    chi2_molly = +(jnp.dot(diff_molly, jnp.linalg.solve(like_dict_0['covariance'], diff_molly)))
    # chi_molly.append(jnp.linalg.solve(jnp.sqrt(like_dict_0['covariance']), diff_molly))
    chi_molly.append(np.multiply(np.diagonal(like_dict_0['covariance']), diff_molly))
    rel_err_molly.append(diff_molly / Y_test[d_i, :] * 100)
    # print(diff_molly/Y_test[d_i,:]<=0)
chi = np.array(chi).T
chi_molly = np.array(chi_molly).T
rel_err = np.array(rel_err).T
rel_err_molly = np.array(rel_err_molly).T
model_linda = np.array(model_linda).T
model_molly = np.array(model_molly).T

rel_err_rms = jnp.sqrt(jnp.mean(jnp.square(rel_err)))
rel_err_std = jnp.std(rel_err)
chi2_dof = chi2 / Y_test.shape[1]
chi2_molly_dof = chi2_molly / Y_test.shape[1]
print(f'chi2 per dof emulator: {chi2_dof},chi2 per dof molly: {chi2_molly_dof}')
print(rel_precision)

#bad_emu=np.append(np.reshape(model_linda[:,np.where(np.min(chi,axis=0)<-2e-9)],[59,8]),np.reshape(model_linda[:,np.where(np.max(chi,axis=0)>2e-9)],[59,9]),axis=1)
#bad_corr=np.append(np.reshape(Y_test.T[:,np.where(np.min(chi,axis=0)<-2e-9)],[59,8]),np.reshape(Y_test.T[:,np.where(np.max(chi,axis=0)>2e-9)],[59,9]),axis=1)
#bad_param=np.append(np.reshape(X_test.T[:,np.where(np.min(chi,axis=0)<-2e-9)],[3,8]),np.reshape(X_test.T[:,np.where(np.max(chi,axis=0)>2e-9)],[3,9]),axis=1)

if __name__ == '__main__':
    '''
    Plot chi and rel_err
    '''
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

    figchi, fig1 = plt.subplots()
    fig1.plot(v_bins, chi, linewidth=0.5, color = 'b', alpha=0.2
            )

    fig1.set_xlabel(r'Velocity [$km s^{-1}$]')
    fig1.set_ylabel(r'$\chi$')
    fig1.legend()

    fig_rel_err, fig2 = plt.subplots()
    fig2.plot(v_bins, rel_err, linewidth=0.5, color = 'b', alpha=0.2
            )
    fig2.set_xlabel(r'Velocity [$km s^{-1}$]')
    fig2.set_ylabel(r'Relative error (%)')
    fig2.set_title(f'rms: {rel_err_rms}; std: {rel_err_std}')
    figchi.savefig(out_path + f'chi_{test_num}.pdf')
    fig_rel_err.savefig(out_path + f'rel_err_{test_num}.pdf')

    '''
    Plot relative error percentiles
    '''
    colormap = cm.Reds
    n = 4
    percentiles = [68,95,99,99.9]
    rel_err_perc= np.zeros((59,n))
    rel_err_molly_perc = np.zeros((59,n))

    for i in range(n):
        rel_err_perc[:,i]=np.percentile(rel_err,percentiles[i],axis=1)
        rel_err_molly_perc[:,i]=np.percentile(abs(rel_err_molly),percentiles[i],axis=1)

    fig_perc, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8,4))
    for i in range(n):
        ax.fill_between(v_bins, rel_err_perc[:,i],color=colormap(i/n),zorder=-i,label=f'{percentiles[i]}%')
    ax.set_title("Percentile plot", fontsize=15)
    ax.tick_params(labelsize=11.5)
    ax.set_xlabel(r'Velocity [$km s^{-1}$]', fontsize=14)
    ax.set_ylabel(r'Relative error Emulator(%)', fontsize=10)
    fig_perc.tight_layout()
    fig_perc.legend()
    fig_perc.savefig(out_path + f'Percentile plot_{test_num}.pdf')

    '''
    Largest chi model
    '''
    '''
    plt.figure(figsize=(10,10))
    plt.plot(v_bins,bad_emu,'r')
    plt.plot(v_bins,bad_corr,'b',ls='-.',alpha=0.5)
    plt.show()
    plt.savefig(out_path + f'largest_chi_model_{test_num}.pdf')

    plt.figure(figsize=(10,10))
    plt.plot(v_bins,(bad_corr-bad_emu)/bad_corr*100,'r')
    plt.ylabel('Relative err [%]')
    plt.show()
    plt.savefig(out_path + f'largest_chi_rel_err_{test_num}.pdf')
    '''

'''
Plot of error in parameter space
'''
def plot_3d_rel_err(rel_err):
    colormap = cm.Reds
    fig3d = plt.figure(figsize=(10, 10))
    ax3d = plt.axes(projection='3d')
    rel_err_rms = jnp.sqrt(jnp.mean(rel_err**2,axis=0))
    for i in range(rel_err_rms.shape[0]):
        ax3d.scatter(X_test[i,0], X_test[i,1], X_test[i,2], color =colormap(rel_err_rms[i]), linewidth=1)
    ax3d.scatter(X_train[:,0], X_train[:,1], X_train[:,2], color = 'b', linewidth=0.5,alpha = 0.1)
    ax3d.set_xlabel(r'$<F>$')
    ax3d.set_ylabel(r'$T_0$')
    ax3d.set_zlabel(r'$\gamma$')
    ax3d.scatter(bad_param[0,:],bad_param[1,:],bad_param[2,:],color = 'b')
    ax3d.view_init(35, 20)
    plt.show()
    fig3d.savefig(out_path + f'param_3d_rel_err_{test_num}.pdf')
