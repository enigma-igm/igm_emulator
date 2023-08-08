import haiku as hk
import jax.numpy as jnp
import jax
from jax.config import config
config.update("jax_enable_x64", True)
from typing import Callable, Iterable, Optional
import optax
import itertools
import struct
import numpy as np
import dill
print(struct.calcsize("P") * 8)


small_bin_bool = True
if small_bin_bool==True:
    #smaller bins
    output_size=[100,100,100,59]
else:
    #larger bins
    output_size=[100,100,100,276]

activation= jax.nn.leaky_relu
#l2 =0.0001
l2 = 0.01
redshift = 5.4 #choose redshift from [5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]
# get the appropriate string and pathlength for chosen redshift
zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
z_idx = np.argmin(np.abs(zs - redshift))
z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
z_string = z_strings[z_idx]
if small_bin_bool == True:
    n_path = 20  # 17->20
    n_covar = 500000
    bin_label = '_set_bins_3'
    in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final_135/{z_string}/'
else:
    n_path = 17
    n_covar = 500000
    bin_label = '_set_bins_4'
    in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'

T0_idx = 8  # 0-14
g_idx = 4  # 0-8
f_idx = 4  # 0-8

like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{g_idx}_SNR0_F{f_idx}_ncovar_{n_covar}_P{n_path}{bin_label}.p'
like_dict = dill.load(open(in_path + like_name, 'rb'))

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
    '''

    Parameters
    ----------
    output_size: list of ints
    activation: activation function
    activate_final: bool
    dropout_rate: float
    name: str

    Returns
    -------

    '''
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

'''
Infrastructure for network training
'''
 ###Learning Rate schedule + Gradient Clipping###
def schedule_lr(lr,total_steps):
    lrate =  optax.piecewise_constant_schedule(init_value=lr,
                                            boundaries_and_scales={int(total_steps*0.2):0.1,
                                                                    int(total_steps*0.4):0.1,
                                                                       int(total_steps*0.6):0.1,
                                                                       int(total_steps*0.8):0.1})
    return lrate

def loss_fn(params, x, y, l2=l2):
    leaves =[]
    for module in sorted(params):
        leaves.append(jnp.asarray(jax.tree_util.tree_leaves(params[module]['w'])))
    regularization =  l2 * sum(jnp.sum(jnp.square(p)) for p in leaves)
    
    diff = custom_forward.apply(params, x) - y
    new_covariance = like_dict['covariance']
    loss = jnp.mean(jnp.abs(diff/jnp.sqrt(jnp.diagonal(new_covariance)))) + regularization
    
    #loss = jnp.mean((custom_forward.apply(params, x) - y) ** 2) + regularization
    return loss

@jax.jit
def accuracy(params, x, y, meanY, stdY):
    preds = custom_forward.apply(params=params, x=x)*stdY+meanY
    y = y*stdY+meanY
    delta = (y - preds) / y
    return delta


def update(params, opt_state, x, y, optimizer):
    print(loss_fn(params, x, y))
    batch_loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, batch_loss, grads
