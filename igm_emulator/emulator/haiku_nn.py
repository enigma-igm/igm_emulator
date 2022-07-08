import jax
import numpy as np
from jax import numpy as jnp
import haiku as hk
from sklearn import datasets
from sklearn.model_selection import train_test_split
from itertools import combinations
import dill
from igm_emulator.scripts.grab_models import *

redshift = 5.4

# get the appropriate string and pathlength for chosen redshift
zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
z_idx = np.argmin(np.abs(zs - redshift))
z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
z_string = z_strings[z_idx]
n_paths = np.array([17, 16, 16, 15, 15, 15, 14])
n_path = n_paths[z_idx]

# read in the parameter grid at given z
param_in_path = '/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/'
param_dict = dill.load(open(param_in_path + f'{z_string}_params.p', 'rb'))

fobs = param_dict['fobs']  # average observed flux <F> ~ Gamma_HI
log_T0s = param_dict['log_T0s']  # log(T_0) from temperature - density relation
T0s = np.exp(log_T0s)
gammas = param_dict['gammas']  # gamma from temperature - density relation


# get the path to the autocorrelation function results from the simulations
in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'
dir = '/home/zhenyujin/igm_emulator/igm_emulator/emulator/LHS'
X = dill.load(open(dir + '/model1.p', 'rb'))
#X, Y = H, models
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=123)

print(X)



