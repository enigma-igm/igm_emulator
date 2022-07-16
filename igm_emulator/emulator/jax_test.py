import jax
import numpy as np
from jax import numpy as jnp
import haiku as hk
from matplotlib import pyplot as plt
from tqdm import trange
import dill
from igm_emulator.scripts.grab_models import *
from jax import value_and_grad
import seaborn as sns

X = np.linspace(-10, 10, num=1000)
Y = 0.1*X*np.cos(X) + 0.1*np.random.normal(size=1000)
X_train, Y_train =  jnp.reshape(jnp.array(X, dtype=jnp.float32),(1000,1)),\
                    jnp.reshape(jnp.array(Y, dtype=jnp.float32),(1000,1))
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

rng = jax.random.PRNGKey(42) ## Reproducibility ## Initializes model with same weights each time.
params = model.init(rng, X_train)
plot_params(params)
epochs = 1000
learning_rate = 0.0001
patience_values = 100
loss = []
best_loss= np.inf
early_stopping_counter = 0

def MeanSquaredErrorLoss(weights, input_data, actual):
    preds = model.apply(weights, rng, input_data)
    preds = preds.squeeze()
    #print(preds.shape,actual.shape)
    return jnp.power(actual - preds, 2).mean()

def UpdateWeights(params,grad):
    return jax.tree_map(lambda p, g: p - g * learning_rate, params, grad)

with trange(epochs) as t:
                for i in t:
                    l, param_grads = value_and_grad(MeanSquaredErrorLoss)(params, X_train, Y_train)

                    params = jax.tree_map(UpdateWeights, params, param_grads)
                    if i % 500 == 0:
                        '''
                        for layer_name, weights in params.items():
                            print(layer_name)
                            print("Weights : {}, Biases : {}\n".format(params[layer_name]["w"],
                                                                       params[layer_name]["b"]))
                        print(f"para_grad:{param_grads}; ")
                        '''

                        plot_params(params)
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

preds = model.apply(params, None, X_train)
#for layer_name, weights in params.items():
    #print(params[layer_name]["w"].shape
print(preds.shape)

fig2, axs = plt.subplots(1,1)
axs.plot(X_train,Y_train,'g',label='real',marker='o',lw=0)
axs.plot(X_train,preds,'r',label='pred')
plt.legend()
plt.show()


