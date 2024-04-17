import jax
import optuna
from optuna.samplers import TPESampler
import sys
import os
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator')
import haiku as hk
from emulator_trainer import TrainerModule
from regulargrid_data import DataSamplerModule
from utils_mlp import DiffStandardScaler
import h5py
import numpy as np
from jax.config import config
import jax.numpy as jnp
import jax
import dill
config.update("jax_enable_x64", True)

# '''
# Set redshift and data bin size
# '''
# redshift = 5.4  # choose redshift from [5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]
# small_bin_bool = True  # True: small bins n=59; False: large bins n=276
#
# '''
# Load datasets
# '''
# # get the appropriate string and pathlength for chosen redshift
# zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
# z_idx = np.argmin(np.abs(zs - redshift))
# z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
# z_string = z_strings[z_idx]
# dir_lhs = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/GRID/'
#
# if small_bin_bool == True:
#     train_num = '_train_55_bin59_seed_11' #'_train_80_bin59_seed_55' #'_train_108_bin59_seed_44' #'_train_68_bin59_seed_33' #'_training_768_bin59' #'_train_100_bin59_seed_42' #'_train_30_bin59_seed_22'   #'_train_300_bin59_seed_66'
#     test_num = '_test_12_bin59_seed_11' #'_test_12_bin59_seed_55' #'_test_15_bin59_seed_44' #'_test_10_bin59_seed_33' #'_test_89_bin59' #'_test_80_bin59_seed_42' #'_test_80_bin59_seed_22'  #_test_80_bin59_seed_66'
#     vali_num = '_vali_45_bin59_seed_11' #'_vali_20_bin59_seed_55' #'_vali_27_bin59_seed_44' #'_vali_18_bin59_seed_33' #'_vali_358_bin59' #'_vali_320_bin59_seed_42' #'_vali_320_bin59_seed_22' #'_vali_320_bin59_seed_66'
#     err_vali_num = '_err_v_221_seed_58_bin59_seed_11' #'_err_v_882_seed_58_bin59_seed_11' #'_err_v_882_bin59_seed_55' #'_err_v_852_bin59_seed_44'
#     n_path = 20  # 17->20
#     n_covar = 500000
#     bin_label = '_set_bins_3'
#     in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final_135/{z_string}/'
#     out_tag = f'{z_string}{train_num}'
# else:
#     train_num = '_train_768'
#     test_num = '_test_89'
#     vali_num = '_vali_358'
#     n_path = 17
#     n_covar = 500000
#     bin_label = '_set_bins_4'
#     in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'
#     out_tag = f'{z_string}{train_num}'
#
# #get the fixed covariance dictionary for likelihood
# T0_idx = 8  # 0-14
# g_idx = 4  # 0-8
# f_idx = 4  # 0-8
# like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{g_idx}_SNR0_F{f_idx}_ncovar_{n_covar}_P{n_path}{bin_label}.p'
# like_dict = dill.load(open(in_path + like_name, 'rb'))
#
# # load the training, test and validation data
# X_og = dill.load(open(dir_lhs + f'{z_string}_param{train_num}.p',
#                    'rb'))  # load normalized cosmological parameters from grab_models.py
# X_test_og = dill.load(open(dir_lhs + f'{z_string}_param{test_num}.p', 'rb'))
# X_vali_og = dill.load(open(dir_lhs + f'{z_string}_param{vali_num}.p', 'rb'))
# x_scaler = DiffStandardScaler(X_og)
# meanX = x_scaler.mean
# stdX = x_scaler.std
# X_train = x_scaler.transform(X_og )
# X_test =  x_scaler.transform(X_test_og )
# X_vali =  x_scaler.transform(X_vali_og )
#
# Y_og = dill.load(open(dir_lhs + f'{z_string}_model{train_num}.p', 'rb'))
# Y_test_og = dill.load(open(dir_lhs + f'{z_string}_model{test_num}.p', 'rb'))
# Y_vali_og = dill.load(open(dir_lhs + f'{z_string}_model{vali_num}.p', 'rb'))
# y_scaler = DiffStandardScaler(Y_og)
# meanY = y_scaler.mean
# stdY = y_scaler.std
# Y_train = y_scaler.transform(Y_og)
# Y_test = y_scaler.transform(Y_test_og)
# Y_vali = y_scaler.transform(Y_vali_og)
#
# # load the NN error covariance and mean
# theta_v = dill.load(open(dir_lhs + f'{z_string}_param{err_vali_num}.p', 'rb'))
# corr_v = dill.load(open(dir_lhs + f'{z_string}_model{err_vali_num}.p', 'rb'))

DataLoader = DataSamplerModule(redshift=5.4,small_bin_bool=True,n_f=3, n_t=6, n_g=3,seed=42,plot_bool=True)   #total_data = 112
X_og, Y_og, X_test_og, Y_test_og, X_vali_og, Y_vali_og, theta_v, corr_v, like_dict = DataLoader.data_sampler()
out_tag = DataLoader.out_tag

# Standardize the data4
x_scaler = DiffStandardScaler(X_og)
meanX = x_scaler.mean
stdX = x_scaler.std
X_train = x_scaler.transform(X_og )
X_test =  x_scaler.transform(X_test_og )
X_vali =  x_scaler.transform(X_vali_og )
y_scaler = DiffStandardScaler(Y_og)

meanY = y_scaler.mean
stdY = y_scaler.std
Y_train = y_scaler.transform(Y_og)
Y_test = y_scaler.transform(Y_test_og)
Y_vali = y_scaler.transform(Y_vali_og)


if __name__ == '__main__':
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
        #bach_size_tune = trial.suggest_categorical('bach_size', [None, 32, 50])
        trainer = TrainerModule(X_train, Y_train, X_test, Y_test, X_vali, Y_vali,
                                x_scaler= x_scaler,
                                y_scaler= y_scaler,
                                layer_sizes= layer_sizes_tune,
                                activation= jax.nn.tanh,#eval(activation_tune),
                                dropout_rate= None,
                                optimizer_hparams=[0.4, lr_tune, 0.003],
                                loss_str= 'mape', #loss_str_tune,
                                loss_weights= [0,0,True],#[l2_tune,c_loss_tune,percent_loss_tune],
                                like_dict=like_dict,
                                init_rng=42,
                                n_epochs= 2000, #n_epochs_tune,
                                pv=100,
                                bach_size= None, #bach_size_tune,
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
                                pv=100,
                                bach_size=hparams['bach_size'],
                                out_tag=out_tag)

        best_param, _ = trainer.train_loop(True)
        dill.dump(best_param, open(
            f'/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/hparam_results/{out_tag}_{trainer.var_tag}_best_param.p',
            'wb'))
        #covar_nn, err_nn = trainer.nn_error_propagation(theta_v, corr_v, save=True, err_vali_num= DataLoader.err_vali_num)
        covar_nn, err_nn = trainer.nn_error_propagation(X_test_og, Y_test_og, save=True, err_vali_num= DataLoader.test_num) #use test data to appro the error
        dill.dump(covar_nn, open(
            f'/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/hparam_results/{out_tag}_{trainer.var_tag}_covar_nn.p',
            'wb'))
        dill.dump(err_nn, open(
            f'/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/hparam_results/{out_tag}_{trainer.var_tag}_err_nn.p',
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
    trial.params['bach_size'] = None
    #trial.params['layer_sizes'] = [100, 59]
    trial.params['activation'] = 'jax.nn.tanh'
    trial.params['n_epochs'] = 2000
    trial.params['max_grad_norm'] = 0.4
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
