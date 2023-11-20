"""igm_emulator - NN emulator for lyman-alpha auto-correlation function.

Straightforward supervised learning (SL) Python library.
It provides all the functions needed to do Neural Network learning and applicationrn
It's backed by the JAX library and the Haiku framework.
"""
from igm_emulator.emulator.emulator_apply import nn_emulator
from igm_emulator.emulator.emulator_trainer import TrainerModule

from igm_emulator.hmc.hmc_nn_inference import NN_HMC_X