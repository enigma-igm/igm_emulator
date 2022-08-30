import haiku as hk
import jax.numpy as jnp
import jax
from typing import Callable, Iterable, Optional
import optax

output_size=[100,100,100,276]
activation= jax.nn.leaky_relu
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

def loss_fn(params, x, y):
  return jnp.mean((custom_forward.apply(params, x) - y) ** 2)

@jax.jit
def accuracy(params, x, y, meanY, stdY):
    preds = custom_forward.apply(params=params, x=x)*stdY+meanY
    y = y*stdY+meanY
    delta = (y - preds) / y
    return delta


def update(params, opt_state, x, y, optimizer):
    batch_loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, batch_loss, grads
