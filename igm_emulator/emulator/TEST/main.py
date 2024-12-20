import dill
import numpy as np
from jax import numpy as jnp
from igm_emulator.emulator.TEST.haiku_nn import haiku_nn

'''
Load Train and Test Data
'''
redshift = 5.4 #choose redshift from
num = '3' #choose data number of LHS sampling
# get the appropriate string and pathlength for chosen redshift
zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
z_idx = np.argmin(np.abs(zs - redshift))
z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
z_string = z_strings[z_idx]
dir_lhs = '/igm_emulator/emulator/LHS/'

X_train = dill.load(open(dir_lhs + f'{z_string}_param{num}.p', 'rb')) # load normalized cosmological parameters from grab_models.py
meanX = X_train.mean(axis=0)
stdX = X_train.std(axis=0)
X_train = (X_train - meanX) / stdX
print(X_train.mean())

Y_train = dill.load(open(dir_lhs + f'{z_string}_model{num}.p', 'rb'))
meanY = Y_train.mean(axis=0)
stdY = Y_train.std(axis=0)
Y_train = (Y_train - meanY) / stdY
print(Y_train.mean())

X_test = dill.load(open(dir_lhs + f'{z_string}_param4.p', 'rb')) # load normalized cosmological parameters from grab_models.py
X_test = (X_test - meanX) / stdX
Y_test = dill.load(open(dir_lhs + f'{z_string}_model4.p', 'rb'))
Y_test = (Y_test- meanY) / stdY
'''
Use Haiku Emulator
'''
data3 = haiku_nn(X_train=jnp.array(X_train, dtype=jnp.float32), Y_train=jnp.array(Y_train, dtype=jnp.float32))
data3.train()
data3.test(X_test=jnp.array(X_test, dtype=jnp.float32),Y_test=jnp.array(Y_test, dtype=jnp.float32))