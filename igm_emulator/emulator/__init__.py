"""emulator - NN emulator for lyman-alpha auto-correlation function.

Straightforward supervised learning (SL) Python library.
It provides all the functions needed to do Neural Network learning and applicationrn
It's backed by the JAX library and the Haiku framework.
"""
from .emulator_run import nn_emulator
from .haiku_custom_forward import small_bin_bool, loss_fn, accuracy, custom_forward
from .emulator_train import train_loop