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
# var_tag = 'huber_l2_1e-05_perc_True_activation_tanh'
'''
trainer = TrainerModule(X_train,Y_train,X_test,Y_test,X_vali,Y_vali,meanX,stdX,meanY,stdY,
                        layer_sizes=[100,100,59],
                        activation= jax.nn.tanh,
                        dropout_rate=None,
                        optimizer_hparams=[0.30000000000000004, 0.0005946616649768666, 0.00013552715097890048],
                        loss_str='huber',
                        loss_weights=[1e-05,0.0033025697025815485,True],
                        like_dict=like_dict,
                        init_rng=42,
                        n_epochs=1000,
                        pv=100,
                        out_tag=out_tag)
                        
'''
###Standard pre-optuna MSE training

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