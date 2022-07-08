import jax
from jax import numpy as jnp
import haiku as hk
from sklearn import datasets
from sklearn.model_selection import train_test_split
'''
load data 
'''
X, Y = datasets.load_boston(return_X_y=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=123)

X_train, X_test, Y_train, Y_test = jnp.array(X_train, dtype=jnp.float32),\
                                   jnp.array(X_test, dtype=jnp.float32),\
                                   jnp.array(Y_train, dtype=jnp.float32),\
                                   jnp.array(Y_test, dtype=jnp.float32),\

samples, features = X_train.shape
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
print(samples, features)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
'''
neural network
'''
def FeedForward(x):
    mlp = hk.nets.MLP(output_sizes=[5,10,15,1])
    return mlp(x)
model = hk.transform(FeedForward)
rng = jax.random.PRNGKey(42)

params = model.init(rng, X_train[:5])

print("Weights Type : {}\n".format(type(params)))

for layer_name, weights in params.items():
    print(layer_name)
    print("Weights : {}, Biases : {}\n".format(params[layer_name]["w"].shape,params[layer_name]["b"].shape))
preds = model.apply(params, rng, X_train)
print(preds[:5])