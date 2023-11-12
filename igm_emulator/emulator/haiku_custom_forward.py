import haiku as hk
import jax.numpy as jnp
import jax
from jax.config import config
import jaxopt
config.update("jax_enable_x64", True)
from typing import Callable, Iterable, Optional
import optax
import itertools
import struct
import numpy as np
from jax.scipy.fft import dct
import dill
print(struct.calcsize("P") * 8)

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

def loss_fn(params, x, y, like_dict, custom_forward, l2, c_loss, loss_str='mse', percent=False):
    '''

    Parameters
    ----------
    params: trained weights and biases
    x: standardized input [fob, T0, gamma]
    y: standardized output [auto-correlation function]
    like_dict: dictionary of mean and covariance of the data
    custom_forward: custom haiku implementation
    l2: l2 regularization weight
    c_loss: loss weight for universal uses
    loss_str: name of the loss function in training
    percent: bool, if True, return the percent difference between prediction and true value

    Returns
    -------
    loss: loss function value to minimize on
    '''
    leaves =[]
    for module in sorted(params):
        leaves.append(jnp.asarray(jax.tree_util.tree_leaves(params[module]['w'])))
    regularization =  l2 * sum(jnp.sum(jnp.square(p)) for p in leaves)

    pred = custom_forward.apply(params, x)
    diff = y - pred
    if percent:
        diff = jnp.divide(diff,y)

    new_covariance = like_dict['covariance']
    mse = jnp.mean((diff) ** 2)
    if loss_str=='mse':
        loss = mse + regularization
    elif loss_str=='chi_one_covariance':
        loss = jnp.mean(jnp.abs(diff / jnp.sqrt(jnp.diagonal(new_covariance)))) + regularization
    elif loss_str=='mse+fft':
        loss = mse + c_loss * jnp.mean((dct(pred) - dct(y)) ** 2) + regularization
    elif loss_str=='huber':
        loss = jnp.sum(jaxopt.loss.huber_loss(y, pred, delta=c_loss*jnp.sqrt(jnp.diagonal(new_covariance)))) + regularization
    elif loss_str=='mae':
        loss = jnp.mean(jnp.abs(diff)) + regularization
    return loss

@jax.jit
def accuracy(params, x, y, meanY, stdY, custom_forward):
    preds = custom_forward.apply(params=params, x=x)*stdY+meanY
    y = y*stdY+meanY
    delta = (y - preds) / y
    return delta


def update(params, opt_state, x, y, optimizer, like_dict, custom_forward, l2, c_loss, loss_str, percent):
    batch_loss, grads = jax.value_and_grad(loss_fn)(params, x, y, like_dict, custom_forward, l2, c_loss, loss_str, percent)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, batch_loss, grads
