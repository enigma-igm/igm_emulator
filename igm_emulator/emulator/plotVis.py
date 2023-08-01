from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import jax.numpy as jnp
import jax
import os
import dill
import h5py
import os
from haiku_custom_forward import small_bin_bool
'''
Visualization of hyperparameters
'''
notes = 'bin59'

zstr = 'z54'
dir_lhs = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/GRID/'
z= f'{zstr}_768_relu_l2+adamw'

if small_bin_bool==True:
    num = '_training_768_bin59'
    skewers_per_data = 20  # 17->20
    n_covar = 500000
    bin_label = '_set_bins_3'
    in_path_molly = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final_135/{zstr}/'
    # change path from f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'

    # get initial grid
    in_name_h5py = f'correlation_temp_fluct_skewers_2000_R_30000_nf_9_dict{bin_label}.hdf5'
    with h5py.File(in_path_molly + in_name_h5py, 'r') as f:
        params = dict(f['params'].attrs.items())
else:
    num = '_training_768'
    in_path_hdf5 = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{zstr}/final_135/'
    R_value = 30000.
    skewers_use = 2000
    n_flux = 9
    bin_label = '_set_bins_4'
    added_label = ''
    temp_param_dict_name = f'correlation_temp_fluct_{added_label}skewers_{skewers_use}_R_{int(R_value)}_nf_{n_flux}_dict_set_bins_4.hdf5'
    with h5py.File(in_path_hdf5 + temp_param_dict_name, 'r') as f:
        params = dict(f['params'].attrs.items())
        
Y = dill.load(open(dir_lhs + f'{zstr}_model{num}.p', 'rb'))
out = Y.shape[1]
v_bins = params['v_bins']
fig = {'legend.fontsize': 16,
       'legend.frameon': False,
       'axes.labelsize': 30,
       'axes.titlesize': 30,
       'figure.titlesize': 38,
       'xtick.labelsize': 25,
       'ytick.labelsize': 25,
       'lines.linewidth': 3,
       'lines.markersize': 2,
       'errorbar.capsize': 3,
       'font.family': 'serif',
       # 'text.usetex': True,
       'xtick.minor.visible': True,
       }
plt.rcParams.update(fig)

def plot_params(params):
  fig1, axs = plt.subplots(ncols=2, nrows=4)
  fig1.tight_layout()
  fig1.set_figwidth(12)
  fig1.set_figheight(6)
  for row, module in enumerate(sorted(params)):
    ax = axs[row][0]
    sns.heatmap(params[module]["w"], cmap="YlGnBu", ax=ax)
    ax.title.set_text(f"{module}/w")

    ax = axs[row][1]
    b = np.expand_dims(params[module]["b"], axis=0)
    sns.heatmap(b, cmap="YlGnBu", ax=ax)
    ax.title.set_text(f"{module}/b")
  plt.show()

def gradientsVis(grads, modelName = None):
    fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(7,5))
    for i, layer in enumerate(sorted(grads)):
        ax[0].scatter(i, grads[layer]['w'].mean())
        ax[0].title.set_text(f'W_grad {modelName}')

        ax[1].scatter(i, grads[layer]['b'].mean())
        ax[1].title.set_text(f'B_grad {modelName}')
    plt.show()
    return fig

def input_overplot(X_train,X_test,X_vali):
    H = X_train
    T = X_test
    V = X_vali
    fig = plt.figure()
    ax = fig.add_subplot(121,projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax.scatter(H[:, 0], H[:, 1], H[:, 2],linewidth=0.5, alpha=0.2)
    ax.scatter(T[:, 0], T[:, 1], T[:, 2], c=T[:, 1], cmap='spring', linewidth=0.5)
    ax2.scatter(H[:, 0], H[:, 1], H[:, 2], linewidth=0.5, alpha=0.2)
    ax2.scatter(V[:, 0], V[:, 1], V[:, 2], c=V[:, 1], cmap='hot', linewidth=0.5)
    ax.set_xlabel(r'$<F>$')
    ax.set_ylabel(r'$T_0$')
    ax.set_zlabel(r'$\gamma$')
    ax.set_title('Test data')
    ax2.set_xlabel(r'$<F>$')
    ax2.set_ylabel(r'$T_0$')
    ax2.set_zlabel(r'$\gamma$')
    ax2.set_title('Validation data')
    fig.suptitle('Parameters space')
    plt.show()

def params_grads_distribution(loss_fn,init_params,X_train,Y_train):
    batch_loss, grads = jax.value_and_grad(loss_fn)(init_params, X_train, Y_train)
    plt.figure(1,figsize=(12, 6))
    sns.distplot(init_params['custom_linear/~/linear_0']['w'])
    sns.distplot(init_params['custom_linear/~/linear_1']['w'])
    sns.distplot(init_params['custom_linear/~/linear_2']['w'])
    plt.legend(labels=['layer1', 'layer2', 'layer3'], title='init_params_w')
    plt.show()

    plt.figure(2, figsize=(18, 4))
    sns.distplot(grads['custom_linear/~/linear_0']['w'])
    sns.distplot(grads['custom_linear/~/linear_1']['w'])
    sns.distplot(grads['custom_linear/~/linear_2']['w'])
    plt.legend(labels=['layer1', 'layer2', 'layer3'], title='grads_w')
    plt.show()

    plt.figure(2, figsize=(18, 4))
    sns.distplot(grads['custom_linear/~/linear_0']['b'])
    sns.distplot(grads['custom_linear/~/linear_1']['b'])
    sns.distplot(grads['custom_linear/~/linear_2']['b'])
    plt.legend(labels=['layer1', 'layer2', 'layer3'], title='grads_b')
    plt.show()

def train_overplot(preds, X, Y, meanY, stdY):
    ax = v_bins # velocity bins
    sample = 8  # number of functions plotted
    fig, axs = plt.subplots(1, 1)
    fig.set_figwidth(15)
    fig.set_figheight(15)
    corr_idx = np.random.randint(0, Y.shape[0], sample)  # randomly select correlation functions to compare
    preds = preds * stdY + meanY
    for i in range(sample):
        axs.plot(ax, preds[corr_idx[i]], label=f'Emulated {i}:' r'$<F>$='f'{X[corr_idx[i], 0]:.2f},'
                                               r'$T_0$='f'{X[corr_idx[i], 1]:.2f},'
                                               r'$\gamma$='f'{X[corr_idx[i], 2]:.2f}', c=f'C{i}', alpha=0.3)
        axs.plot(ax, Y[corr_idx[i]], label=f'Exact {i}', c=f'C{i}', linestyle='--')
    # axs.plot(ax, y_mean, label='Y mean', c='k', alpha=0.2)
    plt.xlabel(r'Velocity/ $km s^{-1}$')
    plt.ylabel('Auto-Correlation')
    plt.title('Train overplot in data space')
    plt.legend()
    dir_exp = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/EXP/'  # plot saving directory
    plt.savefig(os.path.join(dir_exp, f'train_overplot_{z}_{X.shape[0]}_{notes}.png'))
    print('Train overplot saved')
    plt.show()

def test_overplot(test_preds, Y_test, X_test,meanX,stdX,meanY,stdY):
    ax = v_bins
    sample = 10  # number of functions plotted
    fig2, axs2 = plt.subplots(1, 1)
    fig2.set_figwidth(15)
    fig2.set_figheight(15)
    corr_idx = np.random.randint(0, Y_test.shape[0], sample)
    test_preds = test_preds*stdY+meanY
    Y_test = Y_test*stdY+meanY
    X_test = X_test*stdX+meanX
    for i in range(sample):
        axs2.plot(ax, test_preds[corr_idx[i]], label=f'Emulated {i}:' r'$<F>$='f'{X_test[corr_idx[i], 0]:.2f},'
                                                     r'$T_0$='f'{X_test[corr_idx[i], 1]:.2f},'
                                                     r'$\gamma$='f'{X_test[corr_idx[i], 2]:.2f}', c=f'C{i}', alpha=0.3)
        axs2.plot(ax, Y_test[corr_idx[i]], label=f'Exact {i}', c=f'C{i}', linestyle='--')
    # axs.plot(ax, y_mean, label='Y mean', c='k', alpha=0.2)
    plt.xlabel(r'Velocity/ $km s^{-1}$')
    plt.ylabel('Auto-Correlation')
    plt.title('Test overplot in data space')
    plt.legend()
    dir_exp = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/EXP/'  # plot saving directory
    plt.savefig(os.path.join(dir_exp, f'test_overplot_{z}_{X_test.shape[0]}_{notes}.png'))
    plt.show()

def plot_residue(new_delta):
    plt.figure(figsize=(15, 15))
    for i in range(new_delta.shape[0]):
        plt.plot(v_bins, new_delta[i, :] * 100, linewidth=0.5)
    plt.plot(v_bins, jnp.ones([out]), c='r')
    plt.plot(v_bins, -jnp.ones([out]), c='r')
    plt.xlabel(r'Velocity [$km s^{-1}$]')
    plt.ylabel('[Residual] [%]')
    plt.title(f'%Residual plot:mean: {np.mean(new_delta) * 100:.3f}%; std: {np.std(new_delta) * 100:.3f}%')
    dir_exp = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/EXP/'  # plot saving directory
    plt.savefig(os.path.join(dir_exp, f'test%error_{z}_{notes}.png'))
    plt.show()


def bad_learned_plots(delta,X_test,Y_test,test_preds,meanY,stdY):
    unlearnt_idx = []
    for i, d in enumerate(delta):
        for j, e in enumerate(d):
            if e > 5 / 100 or e < -5 / 100: #define the standard of bad leanred (>5%)
                unlearnt_idx.append(i)
                break
    unlearnt_idx = jnp.asarray(unlearnt_idx)
    print(f'unlearned:{unlearnt_idx.shape}')

    ax = v_bins
    fig2, axs2 = plt.subplots(2, 1)
    fig2.set_figwidth(15)
    fig2.set_figheight(30)
    axs2[0].title.set_text('unlearned fitting overplot')
    axs2[1].title.set_text('unlearned residual [%]')
    Y_test = Y_test * stdY + meanY
    test_preds = test_preds * stdY + meanY
    for i in range(unlearnt_idx.shape[0]):
        axs2[0].plot(ax, test_preds[unlearnt_idx[i]], label=f'Preds {i}:' r'$<F>$='f'{X_test[unlearnt_idx[i], 0]:.2f},'
                                                            r'$T_0$='f'{X_test[unlearnt_idx[i], 1]:.2f},'
                                                            r'$\gamma$='f'{X_test[unlearnt_idx[i], 2]:.2f}', c=f'C{i}',
                     alpha=0.3, linestyle='-.')
        axs2[0].plot(ax, Y_test[unlearnt_idx[i]], label=f'Real {i}', c=f'C{i}', linestyle='--')
        axs2[1].plot(ax, delta[unlearnt_idx[i], :] * 100, c=f'C{i}', label=f'%{i}', linewidth=0.6)
    # axs.plot(ax, y_mean, label='Y mean', c='k', alpha=0.2)
    plt.xlabel(r'Velocity/ [$km s^{-1}$]')
    plt.ylabel('Correlation function %')
    plt.title(f'unlearned residue percentage: {unlearnt_idx.shape[0]} sets')
    plt.legend()
    dir_exp = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/EXP/'  # plot saving directory
    plt.savefig(os.path.join(dir_exp, f'unlearnt_{z}_{notes}.png'))
    plt.show()

def plot_error_distribution(new_delta):
    plt.hist(new_delta.flatten(), bins=100, range=[-0.2, 0.2])
    plt.show()
    print(f'%mean: {np.mean(new_delta) * 100}; %std: {np.std(new_delta) * 100}')