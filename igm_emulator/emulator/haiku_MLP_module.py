import types
from typing import Callable, Iterable, Optional
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import dill
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

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

class MLP(hk.Module):
  def __init__(
      self,
      output_sizes: Iterable[int],
      w_init = hk.initializers.Initializer,
      b_init = hk.initializers.Initializer,
      with_bias: bool = True,
      activation = jax.nn.relu,
      activate_final: bool = False,
      name: Optional[str] = None,
  ):

    super().__init__(name=name)
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init
    self.activation = activation
    self.activate_final = activate_final
    layers = []
    output_sizes = tuple(output_sizes)
    for index, output_size in enumerate(output_sizes):
      layers.append(hk.Linear(output_size=output_size,
                              w_init=w_init,
                              b_init=b_init,
                              with_bias=with_bias,
                              name="linear_%d" % index))
    self.layers = tuple(layers)
    self.output_size = output_sizes[-1] if output_sizes else None

  def __call__(
      self,
      inputs: jnp.ndarray,
      dropout_rate: Optional[float] = 0.1,
      rng=None,
  ) -> jnp.ndarray:

    rng = hk.PRNGSequence(rng) if rng is not None else None
    num_layers = len(self.layers)

    out = inputs
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < (num_layers - 1) or self.activate_final:
        # Only perform dropout if we are activating the output.
        if dropout_rate is not None:
          out = hk.dropout(next(rng), dropout_rate, out)
        out = self.activation(out)

    return out

def FeedForward(x):
  model = MLP(
  return model

model = hk.transform(FeedForward, apply_rng=True)
params = model.init(jax.random.PRNGKey(42), x=X_train)
print(params)
'''
def main(X:jnp.ndarray=[],Y:jnp.ndarray=[],lr,epochs,layers):
  params = MLP.init(output_sizes=layers,rng)
  optimizer = optax.adam(lr)
  opt_state = optimizer.init(params)

  def MeanSquaredErrorLoss(params, X, Y):
    compute_loss = jnp.mean((self.model.apply(params, None, X) - Y) ** 2)
    return compute_loss

@jax.jit
  def train(epochs):
    loss = []
    best_loss = np.inf
    early_stopping_counter = 0

    # Training loop
    with trange(epochs) as t:
      for i in t:
        # optimizing loss by gradient descent
        l, grads = value_and_grad(MeanSquaredErrorLoss)(params, x, self.y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        # compute validation loss at the end of the epoch
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
        if early_stopping_counter >= self.pv:
          print('Loss = ' + str(best_loss))
          print('Model saved.')
          break

      print('Reached max number of epochs. Loss = ' + str(best_loss))
      print(f'Model of size {layers} saved.')
    # Final trained parameters and resulting prediction
    params = params
    preds = MLP.apply(params, None, X)

    print(best_loss)

  train()

@jax.jit
  def test(X_test, Y_test):
    test_preds = MLP.apply(params, None, X_test)
    test_loss = MeanSquaredErrorLoss(params, X_test, Y_test)
    test_R2 = r2_score(test_preds.squeeze(), Y_test)

    # Print performance of emulator on test data
    print("Test MSE Loss: {}\n".format(test_loss))  # Loss
    print('Test R^2 Score: {}\n'.format(test_R2))  # R^2 score: ranging 0~1, 1 is good model

    # Plot relative error of all test correlation functions
    delta = (Y_test - test_preds) / Y_test * 100
    ax = np.arange(276)
    for i in range(delta.shape[0]):
      # plt.ylim(-4,)
      plt.plot(ax, delta[i, :], linewidth=0.5)
    plt.xlabel(r'Will be changed to Velocity/ $km s^{-1}$')
    plt.ylabel('% error on Correlation function')
    plt.title(f'Test Loss = {test_loss:.6f}, R^2 Score = {test_R2:.4f}')
    plt.show()
  test()

main()
'''