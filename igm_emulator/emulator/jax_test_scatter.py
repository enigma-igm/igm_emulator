import numpy as np
import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
import haiku as hk
import seaborn as sns
from jax.config import config
config.update("jax_enable_x64", True)
dtype=jnp.float64

xs = np.reshape(np.linspace(-10, 10, num=1000),(1000,1))
ys = 0.1*xs*np.cos(xs) + 0.1*np.random.normal(size=(1000,1))


def plot_params(params):
  fig1, axs = plt.subplots(ncols=2, nrows=3)
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

def FeedForward(x):
    mlp = hk.nets.MLP(output_sizes=[100,100,1])
    return mlp(x)
model = hk.transform(FeedForward)

rng = jax.random.PRNGKey(0) ## Reproducibility ## Initializes model with same weights each time.
params = model.init(rng,xs)

def loss_fn(params, x, y):
  return jnp.mean((model.apply(params, None, x) - y) ** 2)

LEARNING_RATE = 0.0001

@jax.jit
def update(params, x, y):

  grads = jax.grad(loss_fn)(params, x, y)
  # Note that `grads` is a pytree with the same structure as `params`.
  # `jax.grad` is one of the many JAX functions that has
  # built-in support for pytrees.

  # This is handy, because we can apply the SGD update using tree utils:
  return jax.tree_map(
      lambda p, g: p - LEARNING_RATE * g, params, grads
  )

plot_params(params)
for _ in range(1000):
    params = update(params, xs, ys)
    if _ % 500 == 0:
        plot_params(params)


plt.scatter(xs, ys)
plt.scatter(xs, model.apply(params,None,xs), label='Model prediction')
plt.legend()
plt.show()