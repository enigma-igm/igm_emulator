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
config.update("jax_enable_x64", True)
dtype=jnp.float64


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

X_train = dill.load(open(dir_lhs + f'{z_string}_normparam{num}.p', 'rb')) # load normalized cosmological parameters from grab_models.py
Y_train = dill.load(open(dir_lhs + f'{z_string}_model{num}.p', 'rb'))
Y_train = Y_train/((Y_train.max())-(Y_train.min())) #normalize corr functions for better convergence

X_test = dill.load(open(dir_lhs + f'{z_string}_normparam4.p', 'rb')) # load normalized cosmological parameters from grab_models.py
Y_test = dill.load(open(dir_lhs + f'{z_string}_model4.p', 'rb'))
Y_test = Y_test/((Y_test.max())-(Y_test.min()))

'''
Build custom haiku Module
'''
class MyModuleCustom(hk.Module):
  def __init__(self,
               output_size=[100,100,100,276],
               activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
               activate_final: bool = False,
               dropout_rate: Optional[float] = 0.1,
               name='custom_linear'):
    super().__init__(name=name)
    self.activate_final = activate_final
    self.activation = activation
    self.dropout_rate = dropout_rate
    l = []
    for i, layer in enumerate(output_size):
        z =hk.Linear(output_size=layer, name="linear_%d" % i)
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
  module = MyModuleCustom()
  return module(x)

custom_forward = hk.without_apply_rng(hk.transform(_custom_forward_fn))
params = custom_forward.init(rng=5678, x=X_train)
preds = custom_forward.apply(params=params, x=X_train)

'''
Infrastructure for network training
'''

lr = 0.0001
max_grad_norm = 1
n_epochs = 1000
batch_size = 1
n_examples = X_train.shape[0]

 ###Learning Rate schedule + Gradient Clipping###
def make_lr_schedule(warmup_percentage, total_steps):
    def lr_schedule(step):
        percent_complete = step / total_steps
        before_peak = jax.lax.convert_element_type(
            (percent_complete <= warmup_percentage),
            np.float32
        )
        scale = (
            (before_peak * (percent_complete / warmup_percentage) + (1 - before_peak))
            * (1 - percent_complete)
        )
        return scale
    return lr_schedule

total_steps = n_epochs * (n_examples // batch_size)
lr_schedule = make_lr_schedule(warmup_percentage=0.1, total_steps=total_steps)
optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm),
                        optax.scale_by_adam(eps = lr),
                        optax.scale_by_schedule(lr_schedule))
opt_state = optimizer.init(params)

def loss_fn(params, x, y):
  return jnp.mean((custom_forward.apply(params, x) - y) ** 2)

@jax.jit
def update(params, opt_state, x,y):
    batch_loss, grads = jax.value_and_grad(loss_fn)(params, x,y)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, batch_loss

@jax.jit
def accuracy(params, x, y):
    preds = custom_forward.apply(params=params, x=x)
    return (y - preds) / y * 100

### TRAINING loop###
loss = []
best_loss = np.inf
early_stopping_counter = 0
pv = 100

with trange(n_epochs) as t:
        for step in t:
                # optimizing loss by gradient descent
                params, opt_state, batch_loss = update(params, opt_state, X_train, Y_train)
                if step % 500 == 0:
                    plot_params(params)
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
                    print('Loss = ' + str(best_loss))
                    print('Model saved.')
                    break

        print('Reached max number of epochs. Loss = ' + str(best_loss))
        print(f'Model saved.')
        print(f'early_stopping_counter{early_stopping_counter}')
        # Final trained parameters and resulting prediction
        params = params
        preds = custom_forward.apply(params=params, x=X_train)