import jax
import numpy as np
from jax import numpy as jnp
from jax import random
import haiku as hk
from sklearn import datasets

randi_arr = np.ndarray.tolist(np.append(np.random.randint(1, 277, 3),276))
print(randi_arr)
