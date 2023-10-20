import sys
import os
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator')
import haiku as hk
from emulator_train import TrainerModule, X_train,Y_train,X_test,Y_test,X_vali,Y_vali,meanX,stdX,meanY,stdY,out_tag,like_dict
import h5py
import numpy as np
from jax.config import config
import jax.numpy as jnp
import jax
config.update("jax_enable_x64", True)

# Load the archetecture for best parameters after Optuna training
# var_tag = 'chi_one_covariance_l2_1.6e-05_activation_tanh_layers_[100, 100, 59]'
'''
trainer = TrainerModule(X_train,Y_train,X_test,Y_test,X_vali,Y_vali,meanX,stdX,meanY,stdY,
                        layer_sizes=[100,100,59],
                        activation= jax.nn.tanh,
                        dropout_rate=3e-4,
                         optimizer_hparams=[0.05, 6e-4, 2e-4],
                         loss_str='chi_one_covariance',
                         l2_weight=1.6e-5,
                         like_dict=like_dict,
                         init_rng=42,
                         n_epochs=1000,
                         out_tag=out_tag)
'''
max_grad_norm = 0.1
lr = 1e-3
#beta = 1e-3 #BNN
decay = 5e-3
l2 =0.0001

trainer = TrainerModule(X_train,Y_train,X_test,Y_test,X_vali,Y_vali,meanX,stdX,meanY,stdY,
                        layer_sizes=[100,100,100,59],
                        activation= jax.nn.leaky_relu,
                        dropout_rate=None,
                        optimizer_hparams=[max_grad_norm, lr, decay],
                        loss_str='chi_one_covariance',
                        l2_weight=l2,
                        like_dict=like_dict,
                        init_rng=42,
                        n_epochs=1000,
                        pv=100,
                        out_tag=out_tag)


def nn_emulator(best_params_function, theta_linda):
    x = jnp.array((theta_linda - trainer.meanX)/ trainer.stdX)
    emu_out = trainer.custom_forward.apply(best_params_function, x) 
    return emu_out * trainer.stdY + trainer.meanY