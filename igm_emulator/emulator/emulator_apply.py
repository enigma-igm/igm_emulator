import sys
import os
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator')
from emulator_trainer import TrainerModule
from hparam_tuning import X_og,Y_og,X_train,Y_train,X_test,Y_test,X_vali,Y_vali,out_tag,  theta_v, corr_v, like_dict,X_test_og,Y_test_og, x_scaler, y_scaler, DataLoader
from utils_plot import v_bins
import dill
import IPython
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import IPython

### Load best parameters after Optuna training
#var_tag = 'huber_l2_1e-05_perc_True_activation_tanh'
var_tag = 'mape_l2_0_perc_True_activation_tanh' ##shoule automatic implement
#var_tag = 'mape_l2_0_perc_True_activation_sigmoid'
small_bin_bool = DataLoader.small_bin_bool
hparams = dill.load(open(f'/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/hparam_results/{out_tag}_{var_tag}_hparams_tuned.p', 'rb'))
print(out_tag)

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

def _nn_emulator(best_params_function, theta_linda):
    x = jnp.array((theta_linda - trainer.meanX)/ trainer.stdX)
    emu_out = trainer.custom_forward.apply(best_params_function, x) 
    return emu_out * trainer.stdY + trainer.meanY

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
    zstr = 'z55'
    dir_exp = f'/mnt/quasar2/zhenyujin/igm_emulator/emulator/plots/{zstr}/'
    best_params, _ = trainer.train_loop(True)
    dill.dump(best_params, open(
        f'/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/hparam_results/{out_tag}_{var_tag}_best_param.p',
        'wb'))
    print(f'Best Trainer:')
    for key, value in hparams.items():
        print(f'-> {key}: {value}')
    in_path_hdf5 = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/best_params/'
    #best_params = dill.load(open(in_path_hdf5 + f'{trainer.out_tag}_{trainer.var_tag}_best_param.p', 'rb'))  # changed to optuna tuned best param

    ### Error propagation

    ## Load the NN error covariance and mean, while save all the sampels' errors

    covar_nn_err, err_nn_err = trainer.nn_error_propagation(theta_v, corr_v, save=True, err_vali_num =  DataLoader.err_vali_num)
    covar_nn_test, err_nn_test = trainer.nn_error_propagation(X_test_og,Y_test_og, save=True, err_vali_num = DataLoader.test_num)

    '''
    ##Plot test overplot for sanity check if apply correcly

    test_preds = nn_emulator(best_params, X_test_og)
    corr_idx = np.random.randint(0, Y_test_og.shape[0], 10)
    difference = np.subtract(test_preds,Y_test_og)
    rel_diff = np.divide(difference, Y_test_og)
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    for i in range(10):
        ax1.plot(v_bins, test_preds[corr_idx[i]], label=f'Emulated {i}:' r'$<F>$='f'{X_test_og[corr_idx[i], 0]:.2f},'
                                                     r'$T_0$='f'{X_test_og[corr_idx[i], 1]:.2f},'
                                                     r'$\gamma$='f'{X_test_og[corr_idx[i], 2]:.2f}', c=f'C{i}', alpha=0.3)
        ax1.plot(v_bins, Y_test_og[corr_idx[i]], label=f'Exact {i}', c=f'C{i}', linestyle='--')
        ax2.plot(np.array(100 * difference[i, :] / Y_test_og[i, :]).T, color='b', alpha=0.1)
    plt.title(f'mean: {np.mean(rel_diff) * 100:.3f}%; std error: {np.std(rel_diff) * 100:.3f}%')
    plt.savefig(os.path.join(dir_exp,'emulator_apply_test.png'))
    print('emulator_apply_test.png saved')
    '''