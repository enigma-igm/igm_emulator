from matplotlib import pyplot as plt
from matplotlib import ticker
import seaborn as sns
import numpy as np
import jax.numpy as jnp
import jax
import os
import dill
import h5py
from matplotlib import cm
import matplotlib.gridspec as gridspec
import dill
import sys
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator')
from lya_thermal_emulator_setup import redshift

'''
Visualization of hyperparameters
'''

zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
z_idx = np.argmin(np.abs(zs - redshift))
z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
zstr = z_strings[z_idx]

small_bin_bool = True
dir_exp = f'/mnt/quasar2/zhenyujin/igm_emulator/emulator/plots/{zstr}/'

if os.path.exists(dir_exp) is False:
    os.makedirs(dir_exp)

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

x_size = 3.5
dpi_value = 200

plt_params = {'legend.fontsize': 12,
              'legend.frameon': False,
              'axes.labelsize': 12,
              'axes.titlesize': 12,
              'figure.titlesize': 11,
              'xtick.labelsize': 11,
              'ytick.labelsize': 11,
              'lines.linewidth': .7,
              'lines.markersize': 2.3,
              'lines.markeredgewidth': .9,
              'errorbar.capsize': 2,
              'font.family': 'serif',
              # 'text.usetex': True,
              'xtick.minor.visible': True,
            'ytick.minor.visible': True
              }
plt.rcParams.update(plt_params)

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

def train_overplot(preds, X, Y,  meanX,stdX, meanY, stdY, out_tag, var_tag):
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
    sample = 20  # number of functions plotted
    fig, axs = plt.subplots(1, 1)
    fig.set_figwidth(8)
    fig.set_figheight(4)
    corr_idx = np.random.randint(0, Y.shape[0], sample)  # randomly select correlation functions to compare
    preds = preds * stdY + meanY
    Y = Y * stdY + meanY
    X = X * stdX + meanX
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

def test_overplot(test_preds, Y_test, X_test, meanX,stdX,meanY,stdY, out_tag, var_tag,residual_plot = True):
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
    sample = 9  # number of functions plotted
    fig2, axes = plt.subplots(3,3,figsize=(x_size * 3.5 * 0.8, x_size * 2 * 0.8), constrained_layout=True, dpi=dpi_value,sharey='row')
    fig2.set_constrained_layout_pads(
        w_pad=.025, h_pad=.025,
        hspace=0, wspace=0
    )
    axs2_list = np.empty((3, 3), dtype=object)
    if residual_plot:
        new_ax_list = np.empty((3, 3), dtype=object)
        for row in range(3):
            for col in range(3):
                axes[row, col].tick_params(axis='both', which='both', labelbottom=False,labelleft=False, bottom=False, top=False, left=False, right=False)
                axs2 = axes[row, col].inset_axes([0, 0.2, 1, 0.8])  # Top subplot shares x-axis with parent
                new_ax = axes[row, col].inset_axes([0, 0, 1, 0.2], sharex=axs2)
                axs2_list[row, col] = axs2
                new_ax_list[row, col] = new_ax
    else:
       for a in fig2.get_axes():
            axs2_list[row, col] = a

    corr_idx = np.random.choice(np.arange(Y_test.shape[0]), size=9, replace=False)
    err_scales = np.array([2, 2, 2, 5, 5, 10, 10])
    test_preds = test_preds*stdY+meanY
    Y_test = Y_test*stdY+meanY
    X_test = X_test*stdX+meanX

    random_test_preds = test_preds[corr_idx, :]
    random_Y_test = Y_test[corr_idx, :]
    random_X_test = X_test[corr_idx, :]

    # Get the indices that would sort X_test[:,0]
    sort_indices = np.argsort(random_X_test[:, 0])

    # Use these indices to sort X_test, Y_test, and test_preds
    X_test_sorted = random_X_test[sort_indices]
    Y_test_sorted = random_Y_test[sort_indices]
    test_preds_sorted = random_test_preds[sort_indices]
    err_scale = err_scales[z_idx]

    for row in range(3):
        for col in range(3):
            i = 3 * row + col
            axs2 = axs2_list[row, col]
            if residual_plot:
                new_ax = new_ax_list[row, col]
            axs2.tick_params(axis='x', which='both', labelbottom=False, bottom=True, direction='in')
            if row == 2:
                if residual_plot:
                    new_ax.set_xlabel(r'Velocity [$km s^{-1}$]')
                else:
                    axs2.set_xlabel(r'Velocity [$km s^{-1}$]')
            else:
                if residual_plot:
                    shared_ax_x = new_ax_list[2, col]
                    shared_new_ax_y =  new_ax_list[row, 0]
                    new_ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                    new_ax.sharey(shared_new_ax_y)
                else:
                    shared_ax_x = axs2_list[2, col]
                    axs2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                shared_ax_y = axs2_list[row, 0]
                axs2.sharex(shared_ax_x) #[2, col]
                axs2.sharey(shared_ax_y)
            axs2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            axs2.yaxis.major.formatter.set_powerlimits((0, 0))
            if i == 0:
                axs2.plot(ax, Y_test_sorted[i], label=r'$\xi_F$', c='r', lw=1.5)
                axs2.plot(ax, test_preds_sorted[i], label=r'Ly$\alpha$ Emulator', c='b', linestyle='--', lw=1.5)
                axs2.legend(fontsize=7, loc='upper right')
                if residual_plot:
                    new_ax.plot(ax, (Y_test_sorted[i]-test_preds_sorted[i])/Y_test_sorted[i]*100, label='Percentage Residual',alpha = 0.5, c='k')
                    new_ax.legend(fontsize=7, loc='lower right')
                    new_ax.fill_between(np.arange(ax[0]-2000,ax[-1]+2000), -1, 1, color='r', alpha=0.1)
            else:
                axs2.plot(ax, Y_test_sorted[i], c='r', lw=1.5)
                axs2.plot(ax, test_preds_sorted[i], c='b', linestyle='--', lw=1.5)
                if residual_plot:
                    new_ax.plot(ax, (Y_test_sorted[i]-test_preds_sorted[i])/Y_test_sorted[i]*100, alpha = 0.5, c='k')
                    new_ax.fill_between(np.arange(ax[0]-2000,ax[-1]+2000), -1, 1, color='r', alpha=0.1)
            axs2.set_xlim(ax[0]-100, ax[-1]+100)
            new_ax.set_ylim(-err_scale, err_scale)
            if col == 0:
                axs2.set_ylabel(r"$\xi_F$",color='blue')
                axs2.tick_params(axis='y', which='both', colors='blue')
                if residual_plot:
                    new_ax.set_ylabel('[%]')
            else:
                axs2.tick_params(axis='y', which='both', direction='in',labelleft=False, colors='blue')
                if residual_plot:
                    new_ax.tick_params(axis='y', which='both', direction='in', labelleft=False)


            yticks = axs2.get_yticks().tolist()  # get current y-ticks
            yticks.append(axs2.get_ylim()[1])
            axs2.set_yticks(yticks)
            axs2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4,prune='lower'))

            axs2.text(0.2, 0.4,r'$\langle F \rangle$='+f'{X_test_sorted[i, 0]:.4f},'
                    r'$T_0$='f'{X_test_sorted[i, 1]:.0f},'
                    r'$\gamma$='f'{X_test_sorted[i, 2]:.2f}', transform=axs2.transAxes,fontsize=7)
    fig2.savefig(os.path.join(dir_exp, f'test_overplot_{out_tag}_{var_tag}.png'))
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
    colormap = cm.Reds_r
    n = 3
    percentiles = [68, 95, 99]
    rel_err_perc = np.zeros((59, n))
    bias = np.mean(new_delta, axis=0)
     #WHY DO WE NEED THIS ABSOLUTE VALUE?
    rel_err = []
    for i in range(new_delta.shape[0]):
        rel_err.append((jnp.abs(new_delta[i, :]-bias) * 100))
    rel_err = np.array(rel_err).T

    for i in range(n):
        rel_err_perc[:, i] = np.percentile(rel_err, percentiles[i], axis=1)

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8, 4), constrained_layout=True, dpi=dpi_value)
    for i in range(n):
        ax.fill_between(v_bins, rel_err_perc[:, i], color=colormap(i / n), zorder=-i, label=f'{percentiles[i]}%')
    ax.plot(v_bins, bias * 100, c='k', linestyle='--', linewidth=0.5, label='bias')
    ax.set_title(f"Relative Error: {np.mean(new_delta) * 100:.3f}% $\pm$ {np.std(new_delta) * 100:.3f}%", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_xlabel(r'Velocity [$km s^{-1}$]', fontsize=18)
    ax.set_ylabel(r'$\left| \frac{\xi - \xi_{\mathrm{NN}}}{\xi} \right| \times 100$ [%]', fontsize=20)
    ax.text(0.5 ,0.7, f'z = {redshift}', transform=ax.transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle="square", alpha=0.4))
    ax.legend(fontsize=16, loc='upper right')
    fig.tight_layout()
    plt.savefig(os.path.join(dir_exp, f'error_distribution_{out_tag}_{var_tag}.png'))
    plt.show()

def plot_corr_matrix(covar_data,out_tag, name='covar_nn'):
    fig = plt.figure(figsize=(1.05 * x_size, 0.8 * x_size), constrained_layout=True,
                     dpi=dpi_value,
                     )
    axes = fig.add_subplot()
    covar_image = axes.pcolor(v_bins, v_bins, covar_data/np.sqrt(np.outer(np.diag(covar_data),np.diag(covar_data))),
                              cmap='coolwarm',
                              vmin=-1., vmax=1.,
                              rasterized=True)
    axes.set_xlabel('Velocity (km/s)')
    axes.set_ylabel('Velocity (km/s)')
    cbar = fig.colorbar(covar_image)
    tick_locator = ticker.MaxNLocator(nbins=10)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.set_label('Correlation')
    cbar.ax.minorticks_on()
    axes.set_title(r'$T_0$=9149K, $\gamma$=1.352, $\langle F \rangle$=0.0801')
    fig.savefig(os.path.join(dir_exp, f'correlation_matrix_{out_tag}{name}.png'))
    fig.show()

def plot_covar_matrix(covar_data,out_tag, name='covar_nn'):
    fig = plt.figure(figsize=(1.05 * x_size, 0.8 * x_size), constrained_layout=True,
                     dpi=dpi_value,
                     )
    axes = fig.add_subplot()
    covar_image = axes.pcolor(v_bins, v_bins, covar_data,
                              cmap='coolwarm',
                              rasterized=True)
    axes.set_xlabel('Velocity (km/s)')
    axes.set_ylabel('Velocity (km/s)')
    cbar = fig.colorbar(covar_image)
    tick_locator = ticker.MaxNLocator(nbins=10)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.set_label('Covariance')
    cbar.ax.minorticks_on()
    axes.set_title(name)
    fig.savefig(os.path.join(dir_exp, f'covariance_matrix_{out_tag}{name}.png'))
    fig.show()

def plot_covar_frac(covar_nn_test,covar_data,out_tag,name=None):
    fig1 = plt.figure(figsize=(x_size, 0.8*x_size), constrained_layout=True,
                             dpi=dpi_value,
                             )
    axes = fig1.add_subplot()
    tot_covar = covar_data+covar_nn_test
    sig_frac = covar_nn_test/np.sqrt(np.outer(np.diag(tot_covar), np.diag(tot_covar)))
    sig_frac = sig_frac/np.sqrt(np.abs(sig_frac))
    covar_image = axes.pcolor(v_bins, v_bins, sig_frac*100,
                cmap='OrRd',
                rasterized=True)
    axes.set_xlabel('Velocity (km/s)')
    axes.set_ylabel('Velocity (km/s)')
    cbar = fig1.colorbar(covar_image,format='%.1f')
    tick_locator = ticker.MaxNLocator(nbins=10)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.set_label('[%]')
    cbar.ax.minorticks_on()
    axes.set_title('NN Error/Total Noise')
    fig1.savefig(os.path.join(dir_exp, f'covar_frac_{out_tag}{name}.png'))
    fig1.show()


if __name__ == '__main__':
    dill.dump(v_bins,open(f'/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/{zstr}{bin_label}_v_bins.p',
                   'wb'))
