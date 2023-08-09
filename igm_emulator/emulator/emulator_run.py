import sys
import os
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator')
from emulator_train import custom_forward
from haiku_custom_forward import small_bin_bool
import h5py
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)

redshift = 5.4

# get the appropriate string and pathlength for chosen redshift
zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
z_idx = np.argmin(np.abs(zs - redshift))
z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
z_string = z_strings[z_idx]
in_path_hdf5 = os.path.expanduser('~') + f'/igm_emulator/igm_emulator/emulator/best_params/'

if small_bin_bool==True:
    f = h5py.File(in_path_hdf5 + f'z{redshift}_nn_bin59_savefile.hdf5', 'r')
    print(f.keys())
    emu_name = f'{z_string}_best_param_training_768_bin59.p'
else:
    f = h5py.File(in_path_hdf5 + f'z{redshift}_nn_savefile.hdf5', 'r')
    emu_name = f'{z_string}_best_param_training_768.p'

meanX = np.asarray(f['data']['meanX'])
stdX = np.asarray(f['data']['stdX'])
meanY = np.asarray(f['data']['meanY'])
stdY =  np.asarray(f['data']['stdY'])

def nn_emulator(best_params,theta):
    x = (theta - meanX)/ stdX
    model = custom_forward.apply(params=best_params, x=x)
    model = model * stdY + meanY
    return model