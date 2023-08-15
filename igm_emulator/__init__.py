"""igm_emulator - NN emulator for lyman-alpha auto-correlation function.

Straightforward supervised learning (SL) Python library.
It provides all the functions needed to do Neural Network learning and applicationrn
It's backed by the JAX library and the Haiku framework.
"""
from igm_emulator.emulator.emulator_run import nn_emulator
from igm_emulator.emulator.haiku_custom_forward import small_bin_bool, schedule_lr, loss_fn, accuracy, update, output_size, activation, l2, small_bin_bool
#from igm_emulator.emulator.emulator_train import train_loop

from igm_emulator.hmc.nn_hmc_3d_x import NN_HMC_X