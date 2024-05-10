import jax
import optuna
from optuna.samplers import TPESampler
import sys
import os
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator')
import haiku as hk
from emulator_trainer import TrainerModule
from data_loader import DataSamplerModule
from utils_mlp import DiffStandardScaler
from utils_plot import *
import h5py
import numpy as np
from jax.config import config
import jax.numpy as jnp
import jax
import dill
from IPython import embed
config.update("jax_enable_x64", True)
DataLoader = DataSamplerModule(redshift=6.0, small_bin_bool=True,n_f=4, n_t=7, n_g=4, n_testing=0, seed=11, plot_bool=True)   #total_data = 112
X_og, Y_og, X_test_og, Y_test_og, X_vali_og, Y_vali_og, like_dict = DataLoader.data_sampler()
out_tag = DataLoader.out_tag

# Standardize the data4
x_scaler = DiffStandardScaler(X_og)
meanX = x_scaler.mean
stdX = x_scaler.std
X_train = x_scaler.transform(X_og )
X_test =  x_scaler.transform(X_test_og )[:int(np.round(X_train.shape[0]/5)),:] #20% of training data to evaluate test loss in emulator_trainer
X_vali =  x_scaler.transform(X_vali_og )
y_scaler = DiffStandardScaler(Y_og)

meanY = y_scaler.mean
stdY = y_scaler.std
Y_train = y_scaler.transform(Y_og)
Y_test = y_scaler.transform(Y_test_og)[:int(np.round(X_train.shape[0]/5)),:] #20% of training data to evaluate test loss in emulator_trainer
Y_vali = y_scaler.transform(Y_vali_og)


if __name__ == '__main__':
    if DataLoader.redshift >= 5.9:
        early_stop = 500
    else:
        early_stop = 200
    def objective(trial):
        layer_sizes_tune = trial.suggest_categorical('layer_sizes', [ [100, 100, 100, 59], [100, 100, 59], [100, 59]]) # at least three hidden layers
        #activation_tune = trial.suggest_categorical('activation', ['jax.nn.leaky_relu', 'jax.nn.relu', 'jax.nn.sigmoid', 'jax.nn.tanh'])
        #dropout_rate_tune = trial.suggest_categorical('dropout_rate', [None, 0.05, 0.1])
        #max_grad_norm_tune = trial.suggest_float('max_grad_norm', 0, 1, step=0.1)
        lr_tune = trial.suggest_float('lr', 1e-4,1e-1, log=False)
        #decay_tune = trial.suggest_float('decay', 1e-4, 5e-3, log=False)
        #l2_tune = trial.suggest_categorical('l2', [0, 1e-5, 1e-4, 1e-3])
        #c_loss_tune = trial.suggest_float('c_loss', 1e-3, 1, log=True)
        #percent_loss_tune = trial.suggest_categorical('percent', [True, False])
        #n_epochs_tune = trial.suggest_categorical('n_epochs', [1000, 1500, 2000])
        #loss_str_tune = trial.suggest_categorical('loss_str', ['chi_one_covariance', 'mse', 'mse+fft', 'huber', 'mae'])
        bach_size_tune = trial.suggest_categorical('bach_size', [None, 32, 50]) #[None, 10, 20])
        trainer = TrainerModule(X_train, Y_train, X_test, Y_test, X_vali, Y_vali,
                                x_scaler= x_scaler,
                                y_scaler= y_scaler,
                                layer_sizes= layer_sizes_tune,
                                activation= jax.nn.tanh,#eval(activation_tune),
                                dropout_rate= None,
                                optimizer_hparams= [0.4, lr_tune, 0.003],
                                loss_str= 'mape', #loss_str_tune,
                                loss_weights= [0,0,True],#[l2_tune,c_loss_tune,percent_loss_tune],
                                like_dict=like_dict,
                                init_rng=42,
                                n_epochs= 2000, #n_epochs_tune,
                                pv= early_stop,
                                bach_size= bach_size_tune,
                                out_tag=out_tag)

        best_vali_loss = trainer.train_loop(False)[1]
        del trainer
        return best_vali_loss

    def save_best_param_objective(trial):
        hparams = trial.params
        trainer = TrainerModule(X_train, Y_train, X_test, Y_test, X_vali, Y_vali,
                                x_scaler=x_scaler,
                                y_scaler=y_scaler,
                                layer_sizes=hparams['layer_sizes'],
                                activation=eval(hparams['activation']),
                                dropout_rate=hparams['dropout_rate'],
                                optimizer_hparams=[hparams['max_grad_norm'], hparams['lr'], hparams['decay']],
                                loss_str= hparams['loss_str'],
                                loss_weights=hparams['loss_weights'],
                                like_dict=like_dict,
                                init_rng=42,
                                n_epochs=hparams['n_epochs'],
                                pv=early_stop,
                                bach_size=hparams['bach_size'],
                                out_tag=out_tag)
        best_param, _ = trainer.train_loop(True)
        covar_nn_test, err_nn_test, delta_v_test = trainer.nn_error_propagation(X_test_og, Y_test_og, save=True,
                                                                                err_vali_num=DataLoader.test_num)
        dill.dump(best_param, open(
            f'/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/hparam_results/{out_tag}_{trainer.var_tag}_best_param.p',
            'wb'))
        return trainer.var_tag
        del trainer

    print('*** Running the hyperparameter tuning ***')

    # create the study
    number_of_trials = 100
    sampler = TPESampler(seed=10)  # 10
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=number_of_trials, gc_after_trial=True)

    trial = study.best_trial
    #trial.params['bach_size'] = 32
    #trial.params['layer_sizes'] = [100, 59]
    trial.params['activation'] = 'jax.nn.tanh'
    trial.params['n_epochs'] = 2000
    trial.params['max_grad_norm'] =  0.4
    trial.params['decay'] = 0.003
    trial.params['dropout_rate'] = None
    trial.params['loss_str'] = 'mape'
    trial.params['loss_weights'] = [0,0,True] # added fixed strings and weights
    print(f'\nBest Validation Loss: {trial.value}')
    print(f'Best Params:')
    for key, value in trial.params.items():
        print(f'-> {key}: {value}')
    var_tag = save_best_param_objective(trial)
    trial.params['var_tag'] = var_tag

    dill.dump(trial.params, open(f'/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/hparam_results/{out_tag}_{var_tag}_hparams_tuned.p', 'wb'))
    print(f'Best params for optuna tuned emulator saved to /mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/hparam_results/{out_tag}_{var_tag}_hparams_tuned.p')
