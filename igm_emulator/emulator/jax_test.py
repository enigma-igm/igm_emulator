import jax
import numpy as np
from jax import numpy as jnp
import optax
import haiku as hk
from matplotlib import pyplot as plt
from tqdm import trange
import dill
from igm_emulator.scripts.grab_models import *
from jax import value_and_grad
import seaborn as sns
from jax.config import config
config.update("jax_enable_x64", True)
dtype=jnp.float64

'''
X = np.linspace(-10, 10, num=1000)
Y = 0.1*X*np.cos(X) + 0.1*np.random.normal(size=1000)
X_train, Y_train =  jnp.reshape(jnp.array(X, dtype=jnp.float32),(1000,1)),\
                    jnp.reshape(jnp.array(Y, dtype=jnp.float32),(1000,1))
'''

X = np.random.normal(size=(128, 1))
Y = np.reshape(X ** 2,(128, 1))
X_train, Y_train =  jnp.array(X, dtype=jnp.float32),\
                    jnp.array(Y, dtype=jnp.float32)

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
    mlp = hk.nets.MLP(output_sizes=[10,10,1])
    return mlp(x)
model = hk.transform(FeedForward)

rng = jax.random.PRNGKey(42) ## Reproducibility ## Initializes model with same weights each time.
params = model.init(rng, X_train[:5])
#print(params)
epochs = 1000
learning_rate = 0.003
patience_values = 100
loss = []
best_loss= np.inf
early_stopping_counter = 0


def MeanSquaredErrorLoss(params, x, y):
    compute_loss =  jnp.mean((model.apply(params, rng, x) - y) ** 2)
    return compute_loss


optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)
'''
def UpdateWeights(params,grads):
    return jax.tree_util.tree_map(lambda p, g: p - g * learning_rate, params, grads)

'''
'''
#Separate trainable and non-trainable so grad changes accordingly
trainable_params, non_trainable_params = hk.data_structures.partition(
    lambda m, n, p: m != "mlp/~/linear_1" and m != "mlp/~/linear_2" and m != "mlp/~/linear_3", params)
print("trainable:", list(trainable_params))
print("non_trainable:", list(non_trainable_params))
'''

with trange(epochs) as t:
                for i in t:
                    l, grads = value_and_grad(MeanSquaredErrorLoss)(params, X_train, Y_train)
                    updates, opt_state = optimizer.update(grads, opt_state)
                    params = optax.apply_updates(params, updates)
                    if i % 500 == 0:
                        '''
                        for layer_name, weights in params.items():
                            print(layer_name)
                            print("Weights : {}, Biases : {}\n".format(params[layer_name]["w"],
                                                                       params[layer_name]["b"]))
                        print(f"para_grad:{param_grads}; ")
                        '''

                        plot_params(params)
                        print(grads)
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
                    #print (early_stopping_counter)
                    if early_stopping_counter >= patience_values:

                        print('Loss = ' + str(best_loss))
                        print('Model saved.')
                        break

                print('Reached max number of epochs. Loss = ' + str(best_loss))
                print('Model saved.')

preds = model.apply(params, rng, X_train)
#preds = jnp.argmax(model.apply(params, rng, X_train), axis=-1)
#for layer_name, weights in params.items():
    #print(params[layer_name]["w"].shape
print(preds.shape)
print(params)

plt.scatter(X_train,Y_train,label='real')
plt.scatter(X_train,preds,label='pred')
plt.legend()
plt.show()


