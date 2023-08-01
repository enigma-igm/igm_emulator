import haiku as hk
import jax.numpy as jnp
import jax
from jax.config import config
config.update("jax_enable_x64", True)
from typing import Callable, Iterable, Optional
import optax
import itertools
import struct
print(struct.calcsize("P") * 8)


small_bin_bool = True
if small_bin_bool==True:
    #smaller bins
    output_size=[100,100,100,59]
else:
    #larger bins
    output_size=[100,100,100,276]

activation= jax.nn.leaky_relu
l2 =0.0001
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
###Baysian NN add KL divergence
@jax.jit
def gaussian_kl(mu, logvar):
    """Computes mean KL between parameterized Gaussian and Normal distributions.

    Gaussian parameterized by mu and logvar. Mean over the batch.

    NOTE: See Appendix B from VAE paper (Kingma 2014):
          https://arxiv.org/abs/1312.6114
    """
    kl_divergence = jnp.sum(jnp.exp(logvar) + mu**2 - 1 - logvar) / 2
    kl_divergence /= mu.shape[0]

    return kl_divergence

@jax.jit
def sample_params(param, rng):
    def sample_gaussian(mu, logvar):
        eps = jax.random.normal(rng, shape=mu.shape)
        return eps * jnp.exp(logvar / 2) + mu

    sample = jax.tree_multimap(sample_gaussian, sample_gaussian, param['mu'], param['logvar'])
    return sample

def elbo(aprx_posterior, x, y, rng):
    """Computes the Evidence Lower Bound."""
    batch_image, batch_label = batch
    # Sample net parameters from the approximate posterior.
    params = sample_params(aprx_posterior, rng)
    # Get network predictions.
    ogits = net.apply(params, batch_image)
    # Compute log likelihood of batch.
    leaves =[]
    for module in sorted(params):
        leaves.append(jnp.asarray(jax.tree_leaves(params[module]['w'])))
    regularization =  l2 * sum(jnp.sum(jnp.square(p)) for p in leaves)
    # Original negative MSE loss
    log_likelihood = -jnp.mean((custom_forward.apply(params, x) - y) ** 2) + regularization
    # Compute the kl penalty on the approximate posterior.
    kl_divergence = jax.tree_util.tree_reduce(lambda a, b: a + b,
            jax.tree_multimap(gaussian_kl,
                              aprx_posterior['mu'],
                              aprx_posterior['logvar']),
    )
    elbo_ = log_likelihood - 1e-3 * kl_divergence
    return elbo_, log_likelihood, kl_divergence

def loss_fn(params, x, y, rng):
    """Computes the Evidence Lower Bound loss."""
    return -elbo(params, x, y, rng)[0]


###Learning Rate schedule + Gradient Clipping###
def schedule_lr(lr,total_steps):
    lrate =  optax.piecewise_constant_schedule(init_value=lr,
                                            boundaries_and_scales={int(total_steps*0.2):0.1,
                                                                    int(total_steps*0.4):0.1,
                                                                       int(total_steps*0.6):0.1,
                                                                       int(total_steps*0.8):0.1})
    return lrate

@jax.jit
def accuracy(aprx_posterior, x, y, meanY, stdY):
    params = sample_params(aprx_posterior, rng)
    preds = custom_forward.apply(params=params, x=x)*stdY+meanY
    y = y*stdY+meanY
    delta = (y - preds) / y
    return delta


def update(aprx_posterior, opt_state, x, y, rng, optimizer):
    batch_loss, grads = jax.value_and_grad(loss_fn)(params, x, y, rng)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, batch_loss, grads
