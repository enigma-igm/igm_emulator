from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import jax.numpy as jnp
import jax
'''
Visualization of hyperparameters
'''
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

def input_overplot(X_train,X_test):
    H = X_train
    T = X_test
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(H[:, 0], H[:, 1], H[:, 2], c=H[:, 1], cmap='viridis', linewidth=0.5, alpha=0.3)
    ax.scatter(T[:, 0], T[:, 1], T[:, 2], c=T[:, 1], cmap='spring', linewidth=0.5)
    ax.set_xlabel(r'$<F>$')
    ax.set_ylabel(r'$T_0$')
    ax.set_zlabel(r'$\gamma$')
    ax.set_title('Training parameters')
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

def train_overplot(preds, Y_train, X_train):
    ax = np.arange(276)  # arbitrary even spaced x-axis (will be converted to velocityZ)
    sample = 5  # number of functions plotted
    fig, axs = plt.subplots(1, 1)
    fig.set_figwidth(15)
    fig.set_figheight(15)
    corr_idx = np.random.randint(0, Y_train.shape[0], sample)  # randomly select correlation functions to compare
    for i in range(sample):
        axs.plot(ax, preds[corr_idx[i]], label=f'Preds {i}:' r'$<F>$='f'{X_train[corr_idx[i], 0]:.2f},'
                                               r'$T_0$='f'{X_train[corr_idx[i], 1]:.2f},'
                                               r'$\gamma$='f'{X_train[corr_idx[i], 2]:.2f}', c=f'C{i}', alpha=0.3)
        axs.plot(ax, Y_train[corr_idx[i]], label=f'Real {i}', c=f'C{i}', linestyle='--')
    # axs.plot(ax, y_mean, label='Y mean', c='k', alpha=0.2)
    plt.xlabel(r'Will be changed to Velocity/ $km s^{-1}$')
    plt.ylabel('Correlation function')
    plt.legend()
    dir_exp = '/home/zhenyujin/igm_emulator/igm_emulator/emulator/EXP/'  # plot saving directory
    # plt.savefig(os.path.join(dir_exp, f'{self.layers}_overplot{self.comment}.png'))
    plt.show()

def test_overplot(test_preds, Y_test, X_test):
    ax = np.arange(276)
    sample = 5  # number of functions plotted
    fig2, axs2 = plt.subplots(1, 1)
    fig2.set_figwidth(15)
    fig2.set_figheight(15)
    corr_idx = np.random.randint(0, Y_test.shape[0], sample)
    for i in range(sample):
        axs2.plot(ax, test_preds[corr_idx[i]], label=f'Preds {i}:' r'$<F>$='f'{X_test[corr_idx[i], 0]:.2f},'
                                                     r'$T_0$='f'{X_test[corr_idx[i], 1]:.2f},'
                                                     r'$\gamma$='f'{X_test[corr_idx[i], 2]:.2f}', c=f'C{i}', alpha=0.3)
        axs2.plot(ax, Y_test[corr_idx[i]], label=f'Real {i}', c=f'C{i}', linestyle='--')
    # axs.plot(ax, y_mean, label='Y mean', c='k', alpha=0.2)
    plt.xlabel(r'Will be changed to Velocity/ $km s^{-1}$')
    plt.ylabel('Correlation function')
    plt.title('Test overplot')
    plt.legend()
    dir_exp = '/home/zhenyujin/igm_emulator/igm_emulator/emulator/EXP/'  # plot saving directory
    # plt.savefig(os.path.join(dir_exp, f'{self.layers}_overplot{self.comment}.png'))
    plt.show()

def plot_residue(new_delta):
    plt.figure(figsize=(15, 15))
    for i in range(new_delta.shape[0]):
        plt.plot(np.arange(276), new_delta[i, :] * 100, linewidth=0.5)
    plt.plot(np.arange(276), jnp.ones([276]), c='r')
    plt.xlabel(r'Will be changed to Velocity/ $km s^{-1}$')
    plt.ylabel('% error on Correlation function')
    plt.title('Percentage residue plot')
    dir_exp = '/home/zhenyujin/igm_emulator/igm_emulator/emulator/EXP/'  # plot saving directory
    # plt.savefig(os.path.join(dir_exp, f'{self.layers}_test%error{self.comment}.png'))
    plt.show()


def bad_learned_plots(delta,X_test,Y_test,test_preds):
    unlearnt_idx = []
    for i, d in enumerate(delta):
        for j, e in enumerate(d):
            if e > 5 / 100 or e < -5 / 100: #define the standard of bad leanred (>5%)
                unlearnt_idx.append(i)
                break
    unlearnt_idx = jnp.asarray(unlearnt_idx)
    print(f'unlearned:{unlearnt_idx.shape}')

    ax = np.arange(276)
    fig2, axs2 = plt.subplots(2, 1)
    fig2.set_figwidth(15)
    fig2.set_figheight(30)
    axs2[0].title.set_text('unlearned fitting overplot')
    axs2[1].title.set_text('unlearned residue percentage')
    for i in range(unlearnt_idx.shape[0]):
        axs2[0].plot(ax, test_preds[unlearnt_idx[i]], label=f'Preds {i}:' r'$<F>$='f'{X_test[unlearnt_idx[i], 0]:.2f},'
                                                            r'$T_0$='f'{X_test[unlearnt_idx[i], 1]:.2f},'
                                                            r'$\gamma$='f'{X_test[unlearnt_idx[i], 2]:.2f}', c=f'C{i}',
                     alpha=0.3, linestyle='-.')
        axs2[0].plot(ax, Y_test[unlearnt_idx[i]], label=f'Real {i}', c=f'C{i}', linestyle='--')
        axs2[1].plot(ax, delta[unlearnt_idx[i], :] * 100, c=f'C{i}', label=f'%{i}', linewidth=0.6)
    # axs.plot(ax, y_mean, label='Y mean', c='k', alpha=0.2)
    plt.xlabel(r'Will be changed to Velocity/ $km s^{-1}$')
    plt.ylabel('Correlation function %')
    plt.title(f'unleanred {unlearnt_idx.shape[0]}')
    plt.legend()
    dir_exp = '/home/zhenyujin/igm_emulator/igm_emulator/emulator/EXP/'  # plot saving directory
    # plt.savefig(os.path.join(dir_exp, f'{self.layers}_overplot{self.comment}.png'))
    plt.show()

def plot_error_distribution(new_delta):
    plt.hist(new_delta.flatten(), bins=100, range=[-0.2, 0.2])
    plt.show()
    print(f'%mean: {np.mean(new_delta) * 100}; %std: {np.std(new_delta) * 100}')