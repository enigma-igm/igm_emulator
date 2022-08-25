import dill
import numpy as np
import haiku as hk
import jax.numpy as jnp
import jax
from typing import Callable, Iterable, Optional
import optax
from tqdm import trange
from matplotlib import pyplot as plt
import seaborn as sns
from jax.config import config
from sklearn.metrics import r2_score
import random
#from hyperparam_tuner import *


max_grad_norm = 1e-3
n_epochs = 1000
lr = 1e-3
decay = 5e-3
output_size=[500,500,500,276]
activation= jax.nn.sigmoid

config.update("jax_enable_x64", True)
dtype=jnp.float64
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

'''
Load Train and Test Data
'''
redshift = 5.4 #choose redshift from
num = '6_1000' #choose data number of LHS sampling
# get the appropriate string and pathlength for chosen redshift
zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
z_idx = np.argmin(np.abs(zs - redshift))
z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
z_string = z_strings[z_idx]
dir_lhs = '/home/zhenyujin/igm_emulator/igm_emulator/emulator/LHS/'

X_train = dill.load(open(dir_lhs + f'{z_string}_param{num}.p', 'rb')) # load normalized cosmological parameters from grab_models.py
meanX = X_train.mean(axis=0)
stdX = X_train.std(axis=0)
X_train = (X_train - meanX) / stdX
print(X_train.mean())

Y_train = dill.load(open(dir_lhs + f'{z_string}_model{num}.p', 'rb'))
meanY = Y_train.mean(axis=0)
stdY = Y_train.std(axis=0)
Y_train = (Y_train - meanY) / stdY
print(Y_train.mean())

X_test = dill.load(open(dir_lhs + f'{z_string}_param4.p', 'rb')) # load normalized cosmological parameters from grab_models.py
X_test = (X_test - meanX) / stdX
Y_test = dill.load(open(dir_lhs + f'{z_string}_model4.p', 'rb'))
Y_test = (Y_test- meanY) / stdY

'''
Build custom haiku Module
'''
class MyModuleCustom(hk.Module):
  def __init__(self,
               output_size=[100,100,100,276],
               activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
               activate_final: bool = False,
               dropout_rate: Optional[float] = None,
               name='custom_linear'):
    super().__init__(name=name)
    self.activate_final = activate_final
    self.activation = activation
    self.dropout_rate = dropout_rate
    l = []
    for i, layer in enumerate(output_size):
        z =hk.Linear(output_size=layer, name="linear_%d" % i, w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal"))
        l.append(z)
    self.layer = l

  def __call__(self, x, rng = 42):
    num_layers = len(self.layer)
    out = x
    rng = hk.PRNGSequence(rng) if rng is not None else None

    for i, layer in enumerate(self.layer):
        out = layer(out)

        if i < (num_layers - 1) or self.activate_final:
            # Only perform dropout if we are activating the output.
            if self.dropout_rate is not None:
                out = hk.dropout(next(rng), self.dropout_rate, out)
            out = self.activation(out)

    return out

def _custom_forward_fn(x):
  module = MyModuleCustom(output_size=output_size, activation = activation)
  return module(x)

custom_forward = hk.without_apply_rng(hk.transform(_custom_forward_fn))
rng_sequence = hk.PRNGSequence(42)
init_params = custom_forward.init(rng=next(rng_sequence), x=X_train)
preds = custom_forward.apply(params=init_params, x=X_train)

'''
Infrastructure for network training
'''
n_samples = X_train.shape[0]

 ###Learning Rate schedule + Gradient Clipping###
@jax.jit
def schedule_lr(step):
    lrate = lr * jnp.exp(-decay * step)
    return lrate

def loss_fn(params, x, y):
  return jnp.mean((custom_forward.apply(params, x) - y) ** 2)

@jax.jit
def accuracy(params, x, y):
    preds = custom_forward.apply(params=params, x=x)
    delta = jnp.absolute((y - preds) / y) * 100
    return delta

early_stopping_counter = 0
pv = 100

optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm),
                        optax.adam(schedule_lr)
                        )
opt_state = optimizer.init(init_params)

@jax.jit
def update(params, opt_state, x, y):
    batch_loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, batch_loss, grads

### Random batches of size 200###
idx = [np.random.randint(0,1000,1) for i in range(200)]
### TRAINING loop###
best_loss = np.inf
loss = []
params = init_params

if __name__ == "__main__":
    with trange(n_epochs) as t:
        for step in t:
                # optimizing loss by update function
            params, opt_state, batch_loss, grads = update(params, opt_state, X_train, Y_train)

            #if step % 500 == 0:
                    #plot_params(params)
                    #gradientsVis(grads, f'Epoch{step}')
                    #print(f'grads: {grads}')

                # compute validation loss at the end of the epoch
            l = batch_loss
            loss.append(l)

                # update the progressbar
            t.set_postfix(loss=loss[-1])

                # early stopping condition
            if l <= best_loss:
                    best_loss = l
                    early_stopping_counter = 0
            else:
                    early_stopping_counter += 1
                # print (early_stopping_counter)
            if early_stopping_counter >= pv:
                    break

    print('Reached max number of epochs in this batch. Loss = ' + str(best_loss))
    print(f'Model saved.')
    print(f'early_stopping_counter: {early_stopping_counter}')
    print(f'accuracy: {jnp.mean(accuracy(params, X_test, Y_test)[np.random.randint(0,100,20)])}')
    print(f'Test Loss: {loss_fn(params, X_test, Y_test)}')
    #plt.plot(range(len(loss)),loss)
    #plt.show()
        # Final trained parameters and resulting prediction
    preds = custom_forward.apply(params=params, x=X_train)

    # Plot partial predited corrolation functions and
    ax = np.arange(276)  # arbitrary even spaced x-axis (will be converted to velocityZ)
    sample = 5  # number of functions plotted
    fig, axs = plt.subplots(1, 1)
    corr_idx = np.random.randint(0, 100, sample)  # randomly select correlation functions to compare
    for i in range(sample):
        axs.plot(ax, preds[corr_idx[i]], label=f'Preds {i}:' r'$<F>$='f'{X_train[corr_idx[i], 0]:.2f},'
                                               r'$T_0$='f'{X_train[corr_idx[i], 1]:.2f},'
                                               r'$\gamma$='f'{X_train[corr_idx[i], 2]:.2f}', c=f'C{i}', alpha=0.3)
        axs.plot(ax, Y_train[corr_idx[i]], label=f'Real {i}', c=f'C{i}', linestyle='--')
    # axs.plot(ax, y_mean, label='Y mean', c='k', alpha=0.2)
    plt.xlabel(r'Will be changed to Velocity/ $km s^{-1}$')
    plt.ylabel('Correlation function')
    plt.title(f'Train Loss = {best_loss}')
    plt.legend()
    dir_exp = '/home/zhenyujin/igm_emulator/igm_emulator/emulator/EXP/'  # plot saving directory
    # plt.savefig(os.path.join(dir_exp, f'{self.layers}_overplot{self.comment}.png'))
    plt.show()

    test_preds = custom_forward.apply(params, X_test)
    test_loss = loss_fn(params, X_test, Y_test)
    test_R2 = r2_score(test_preds.squeeze(), Y_test)

    # Plot relative error of all test correlation functions
    delta = accuracy(params, X_test, Y_test)
    print(type(delta))

    for i, d in enumerate(delta):
        for j, e in enumerate(d):
            if e > 100:
                jax.lax.dynamic_update_slice(delta,0,(i,j))


    for i in range(delta.shape[0]):
        plt.plot(np.arange(276),delta[i,:], linewidth=0.5)
    plt.xlabel(r'Will be changed to Velocity/ $km s^{-1}$')
    plt.ylabel('% error on Correlation function')
    plt.title(f'Test Loss = {test_loss:.6f}, R^2 Score = {test_R2:.4f}, lr={lr}, decay={decay}')
    dir_exp = '/home/zhenyujin/igm_emulator/igm_emulator/emulator/EXP/'  # plot saving directory
#plt.savefig(os.path.join(dir_exp, f'{self.layers}_test%error{self.comment}.png'))
    plt.show()

    print("Test MSE Loss: {}\n".format(test_loss)) # Loss
    print('Test R^2 Score: {}\n'.format(test_R2))  # R^2 score: ranging 0~1, 1 is good model
