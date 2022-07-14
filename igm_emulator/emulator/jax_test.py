import jax
import numpy as np
from jax import numpy as jnp
import haiku as hk
from matplotlib import pyplot as plt
from tqdm import trange
import dill
from igm_emulator.scripts.grab_models import *
from jax import value_and_grad

X = np.linspace(-10, 10, num=1000)
Y = 0.1*X*np.cos(X) + 0.1*np.random.normal(size=1000)
X_train, Y_train =  jnp.array(np.transpose(X), dtype=jnp.float32),\
                    jnp.array(np.transpose(Y), dtype=jnp.float32),\

def FeedForward(x):
    mlp = hk.nets.MLP(output_sizes=[64,64,1])
    return mlp(x)
model = hk.transform(FeedForward)

rng = jax.random.PRNGKey(42) ## Reproducibility ## Initializes model with same weights each time.
params = model.init(rng, X_train)
epochs = 100
learning_rate = jnp.array(0.001)
patience_values = 100
loss = []
best_loss= np.inf
early_stopping_counter = 0

def MeanSquaredErrorLoss(weights, input_data, actual):
    preds = model.apply(weights, rng, input_data)
    preds = preds.squeeze()
    #print(preds.shape,actual.shape)
    return jnp.power(actual - preds, 2).mean()

def UpdateWeights(weights,gradients):
    return weights - learning_rate * gradients

with trange(epochs) as t:
                for i in t:
                    l, param_grads = value_and_grad(MeanSquaredErrorLoss)(params, X_train, Y_train)
                    params = jax.tree_map(UpdateWeights, params, param_grads)
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
#print(preds[1,:]-Y[1,:],preds[2,:]-Y[2,:])


fig, axs = plt.subplots(1,1)
axs.plot(X_train,Y_train,'g',label='real',marker='o', ms=2)
axs.plot(X_train,preds,'r',label='pred')
plt.legend()
plt.show()


