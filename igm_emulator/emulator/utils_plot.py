from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import jax.numpy as jnp
import jax
import os
import dill
import h5py
import os
from matplotlib import cm

'''
Visualization of hyperparameters
'''

zstr = 'z54'
small_bin_bool = True
dir_exp = f'/mnt/quasar2/zhenyujin/igm_emulator/emulator/plots/{zstr}/'

if small_bin_bool==True:
    skewers_per_data = 20  # 17->20
    n_covar = 500000
    bin_label = '_set_bins_3'
    in_path_molly = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final_135/{zstr}/'
    in_name_h5py = f'correlation_temp_fluct_skewers_2000_R_30000_nf_9_dict{bin_label}.hdf5'

else:
    R_value = 30000.
    skewers_use = 2000
    n_flux = 9
    bin_label = '_set_bins_4'
    added_label = ''
    in_path_molly = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{zstr}/final_135/'
    in_name_h5py = f'correlation_temp_fluct_{added_label}skewers_{skewers_use}_R_{int(R_value)}_nf_{n_flux}_dict_set_bins_4.hdf5'

with h5py.File(in_path_molly + in_name_h5py, 'r') as f:
    params = dict(f['params'].attrs.items())

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

def train_overplot(preds, X, Y, meanY, stdY, out_tag, var_tag):
    '''

    Parameters
    ----------
    preds: standardized prediction
    X
    Y
    meanY
    stdY
    out_tag
    var_tag

    Returns
    -------

    '''
    ax = v_bins # velocity bins
    sample = 8  # number of functions plotted
    fig, axs = plt.subplots(1, 1)
    fig.set_figwidth(15)
    fig.set_figheight(15)
    corr_idx = np.random.randint(0, Y.shape[0], sample)  # randomly select correlation functions to compare
    preds = preds * stdY + meanY
    Y = Y * stdY + meanY
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
    plt.savefig(os.path.join(dir_exp, f'train_overplot_{out_tag}_{var_tag}.png'))
    print('Train overplot saved')
    plt.show()

def test_overplot(test_preds, Y_test, X_test,meanX,stdX,meanY,stdY, out_tag, var_tag):
    '''


    Parameters
    ----------
    test_preds: standardized prediction
    Y_test
    X_test
    meanX
    stdX
    meanY
    stdY
    out_tag
    var_tag

    Returns
    -------

    '''
    ax = v_bins
    sample = 10  # number of functions plotted
    fig2, axs2 = plt.subplots(1, 1)
    fig2.set_figwidth(15)
    fig2.set_figheight(8)
    corr_idx = np.random.randint(0, Y_test.shape[0], sample)
    test_preds = test_preds*stdY+meanY
    Y_test = Y_test*stdY+meanY
    X_test = X_test*stdX+meanX
    for i in range(sample):
        axs2.plot(ax, test_preds[corr_idx[i]], label=f'Emulation {i}:' r'$<F>$='f'{X_test[corr_idx[i], 0]:.2f},'
                                                     r'$T_0$='f'{X_test[corr_idx[i], 1]:.2f},'
                                                     r'$\gamma$='f'{X_test[corr_idx[i], 2]:.2f}', c=f'C{i}', alpha=0.3)
        axs2.plot(ax, Y_test[corr_idx[i]], label=f'Data {i}', c=f'C{i}', linestyle='--')
    # axs.plot(ax, y_mean, label='Y mean', c='k', alpha=0.2)
    plt.xlabel(r'Velocity/ $km s^{-1}$')
    plt.ylabel('Auto-Correlation')
    plt.title('Test data set overplot')
    plt.legend(fontsize=10, loc='upper right', ncol=2
    plt.savefig(os.path.join(dir_exp, f'test_overplot_{out_tag}_{var_tag}.png'))
    print('Test overplot saved')
    plt.show()

def plot_residue(new_delta, out_tag, var_tag):
    plt.figure(figsize=(15, 15))
    for i in range(new_delta.shape[0]):
        plt.plot(v_bins, new_delta[i, :], linewidth=0.5,color = 'b', alpha=0.2)
    plt.plot(v_bins, jnp.ones_like(v_bins), c='r')
    plt.plot(v_bins, -1*jnp.ones_like(v_bins), c='r')
    plt.xlabel(r'Velocity [$km s^{-1}$]')
    plt.ylabel('[Residual/Chi]')
    plt.title(f'Residual/Chi plot:mean: {np.mean(new_delta):.3f}; std: {np.std(new_delta):.3f}')
    plt.savefig(os.path.join(dir_exp, f'test_chi_error_{out_tag}_{var_tag}.png'))
    print('Chi saved')
    plt.show()


def bad_learned_plots(delta,X_test,Y_test,test_preds,meanY,stdY, out_tag, var_tag):
    unlearnt_idx = []
    for i, d in enumerate(delta):
        for j, e in enumerate(d):
            if e > 5 / 100 or e < -5 / 100: #define the standard of bad leanred (>5%)
                unlearnt_idx.append(i)
                break
    unlearnt_idx = jnp.asarray(unlearnt_idx)

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
    plt.savefig(os.path.join(dir_exp, f'unlearnt_{out_tag}_{var_tag}.png'))
    plt.show()

def plot_error_distribution(new_delta,out_tag, var_tag):
    colormap = cm.Reds
    n = 3
    percentiles = [68, 95, 99]
    rel_err_perc = np.zeros((59, n))
    rel_err = []
    for i in range(new_delta.shape[0]):
        rel_err.append(jnp.abs(new_delta[i, :]) * 100)
    rel_err = np.array(rel_err).T
    for i in range(n):
        rel_err_perc[:, i] = np.percentile(rel_err, percentiles[i], axis=1)

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8, 4))
    for i in range(n):
        ax.fill_between(v_bins, rel_err_perc[:, i], color=colormap(i / n), zorder=-i, label=f'{percentiles[i]}%')
    ax.set_title(f"mean error: {np.mean(new_delta) * 100:.3f}%; std error: {np.std(new_delta) * 100:.3f}%", fontsize=15)
    ax.tick_params(labelsize=11.5)
    ax.set_xlabel(r'Velocity [$km s^{-1}$]', fontsize=14)
    ax.set_ylabel(r'Relative error Emulator(%)', fontsize=10)
    fig.tight_layout()
    fig.legend()
    plt.savefig(os.path.join(dir_exp, f'error_distribution_{out_tag}_{var_tag}.png'))
    plt.show()