import sys
import os
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator')
from emulator_trainer import TrainerModule
from hparam_tuning import X_og,Y_og,X_train,Y_train,X_test,Y_test,X_vali,Y_vali,out_tag, like_dict,X_test_og,Y_test_og, x_scaler, y_scaler, DataLoader
from utils_plot import *
import dill
import IPython
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import IPython

'''
Load best parameters after Optuna training
'''
#var_tag = 'huber_l2_1e-05_perc_True_activation_tanh'
var_tag = 'mape_l2_0_perc_True_activation_tanh' ##shoule automatic implement
#var_tag = 'mape_l2_0_perc_True_activation_sigmoid'
small_bin_bool = DataLoader.small_bin_bool
test_num = DataLoader.test_num
z_string = DataLoader.z_string
redshift = DataLoader.redshift
if redshift >= 5.9:
    early_stop = 500
else:
    early_stop = 200

hparams = dill.load(open(f'/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/hparam_results/{out_tag}_{var_tag}_hparams_tuned.p', 'rb'))
print(f'Emulator in use: {out_tag}_{var_tag}')

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

def _nn_emulator(best_params_function, theta_linda):
    x = trainer.x_scaler.transform(theta_linda)
    emu_out = trainer.custom_forward.apply(best_params_function, x) 
    return trainer.y_scaler.inverse_transform(emu_out)

def nn_emulator(best_params_function, theta_linda):
    '''
    give emulator prediction for multiple sets of [fob, T0, gamma]
    '''
    emu_out = jax.vmap(_nn_emulator, in_axes=(None,0), out_axes=0)(best_params_function, jnp.atleast_2d(theta_linda))

    return emu_out.squeeze()

'''
Check if jvmap works
'''
if __name__ == '__main__':
    dir_exp = f'/mnt/quasar2/zhenyujin/igm_emulator/emulator/plots/{z_string}/'
    in_path_best_params = '/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/hparam_results/'

    ## Don't have to retrain each time -- if want training plots then fine
    best_params, _ = trainer.train_loop(True)
    #best_params = dill.load(open(in_path_best_params + f'{out_tag}_{var_tag}_best_param.p','rb'))

    print(f'Best Trainer:')
    for key, value in hparams.items():
        print(f'-> {key}: {value}')
    in_path_hdf5 = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/best_params/'

    ## Test overplot (including err prop points) without retraining
    test_preds = trainer.custom_forward.apply(best_params, trainer.X_test)
    #test_overplot(test_preds, trainer.Y_test, trainer.X_test, trainer.meanX, trainer.stdX, trainer.meanY, trainer.stdY, trainer.out_tag,
                  #trainer.var_tag)

    ### Error propagation

    ## Load the NN error covariance and mean, while save all the sampels' errors
    covar_nn_test, err_nn_test, delta_v_test = trainer.nn_error_propagation(X_test_og,Y_test_og, save=True, err_vali_num = DataLoader.test_num)
    covar_data = trainer.like_dict['covariance']
    
    dill.dump(covar_nn_test, open(
            f'/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/hparam_results/{out_tag}_{trainer.var_tag}_covar_nn.p',
            'wb'))
    dill.dump(err_nn_test, open(
            f'/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/hparam_results/{out_tag}_{trainer.var_tag}_err_nn.p',
            'wb'))
   

    neg_count_test = 0
    for i in covar_nn_test.flatten():
        if i < 0:
            neg_count_test += 1
    print(f'Negative count in error propagation for {DataLoader.test_num}: {neg_count_test}')

    ## Plot the error propagation results
    plt.figure(figsize=(12, 6))
    plt.plot(v_bins, delta_v_test.T, color='b', alpha=0.1)
    plt.plot(v_bins, err_nn_test, color='r', label='mean')
    plt.title(f'Error propagation {DataLoader.test_num}')
    plt.savefig(os.path.join(dir_exp, f'error_propagation_{DataLoader.test_num}.png'))
    plt.show()
    plt.close()

    plot_corr_matrix(covar_nn_test, out_tag=out_tag, name=f'covar_nn_{DataLoader.test_num}')
    plot_corr_matrix(covar_data, out_tag=out_tag, name='covar_data')
    plot_covar_matrix(covar_nn_test, out_tag=out_tag, name=f'covar_nn_{DataLoader.test_num}')
    plot_covar_matrix(covar_data, out_tag=out_tag, name='covar_data')
    plot_covar_frac(covar_nn_test, covar_data, out_tag=out_tag, name=DataLoader.test_num)

