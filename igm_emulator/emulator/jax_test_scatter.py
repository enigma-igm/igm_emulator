import numpy as np
import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
import haiku as hk
import seaborn as sns
from jax.config import config
config.update("jax_enable_x64", True)
dtype=jnp.float64
'''
xs = np.reshape(np.linspace(-10, 10, num=1000),(1000,1))
ys = 0.1*xs*np.cos(xs) + 0.1*np.random.normal(size=(1000,1))
'''
X_train = np.random.normal(size=(128, 1))
Y_train = np.reshape(X_train ** 2,(128, 1))

def init_mlp_params(layer_widths):
  params = []
  for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
    params.append(
        dict(weights=np.random.normal(size=(n_in, n_out)) * np.sqrt(2/n_in),
             biases=np.zeros(shape=(n_out,))
            )
    )
  return params

params = init_mlp_params([1, 10, 10, 1])


def forward(params, x):
  *hidden, last = params
  for layer in hidden:
    x = jax.nn.relu(x @ layer['weights'] + layer['biases'])
  return x @ last['weights'] + last['biases']

def loss_fn(params, x, y):
  return jnp.mean((forward(params, x) - y) ** 2)

LEARNING_RATE = 0.001

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

def plot_params(params):
  fig1, axs = plt.subplots(ncols=2, nrows=3)
  fig1.tight_layout()
  fig1.set_figwidth(12)
  fig1.set_figheight(6)
  *hidden, last = params
  print(f'hidden:{hidden}')
  for row,layer in enumerate(hidden):
    ax = axs[row][0]

    sns.heatmap(layer['weights'], cmap="YlGnBu", ax=ax)
    ax.title.set_text(f"{row}/w")

    ax = axs[row][1]
    b = np.expand_dims(layer["biases"], axis=0)
    sns.heatmap(b, cmap="YlGnBu", ax=ax)
    ax.title.set_text(f"{row}/b")
  ax = axs[2][0]
  sns.heatmap(layer['weights'], cmap="YlGnBu", ax=ax)
  ax.title.set_text("2/w")

  ax = axs[2][1]
  b = np.expand_dims(last["biases"], axis=0)
  sns.heatmap(b, cmap="YlGnBu", ax=ax)
  ax.title.set_text("2/b")
  plt.show()

plot_params(params)
for _ in range(2000):
    params = update(params, xs, ys)
    if _ % 500 == 0:
        plot_params(params)

plot_params(params)

plt.scatter(xs, ys)
plt.scatter(xs, forward(params, xs), label='Model prediction')
plt.legend()
plt.show()