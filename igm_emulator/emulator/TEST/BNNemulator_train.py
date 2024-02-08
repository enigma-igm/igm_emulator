import dill
import os
import numpy as np
import haiku as hk
import jax.numpy as jnp
import jax
from typing import Callable, Iterable, Optional
import optax
from tqdm import trange
from jax.config import config
from sklearn.metrics import r2_score
import sys
import os
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator')
from utils_plot import *
from bayes import _bnn_custom_forward_fn, schedule_lr, elbo, loss_fn, accuracy, update, output_size, activation, l2, predict, small_bin_bool
import h5py
import IPython

max_grad_norm = 0.1
n_epochs = 1000
lr = 1e-3
beta = 1e-3
decay = 5e-3
my_rng = jax.random.PRNGKey(42)
print(f'Training for small bin: {small_bin_bool}')
print(f'Layers: {output_size}')
print(f'Activation: {activation}')
print(f'L2 regularization lambda: {l2}')
config.update("jax_enable_x64", True)
dtype=jnp.float64

'''
Load Train and Test Data
'''
redshift = 5.4 #choose redshift from [5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]


# get the appropriate string and pathlength for chosen redshift
zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
z_idx = np.argmin(np.abs(zs - redshift))
z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
z_string = z_strings[z_idx]
dir_lhs = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/GRID/'

if small_bin_bool==True:
    train_num = '_training_768_bin59'
    test_num = '_test_89_bin59'
    vali_num = '_vali_358_bin59'
    n_path = 20  # 17->20
    n_covar = 500000
    bin_label = '_set_bins_3'
    in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final_135/{z_string}/'
else:
    train_num = '_training_768'
    test_num = '_test_89'
    vali_num = '_vali_358'
    n_path = 17
    n_covar = 500000
    bin_label = '_set_bins_4'
    in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'
out_tag = f'{z_string}{train_num}'
var_tag = 'BNN_elbo_activation_relu'

T0_idx = 8 #0-14
g_idx = 4 #0-8
f_idx = 4 #0-8
   
like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{g_idx}_SNR0_F{f_idx}_ncovar_{n_covar}_P{n_path}{bin_label}.p'
like_dict = dill.load(open(in_path + like_name, 'rb'))

in_name_h5py = f'correlation_temp_fluct_skewers_2000_R_30000_nf_9_dict{bin_label}.hdf5'
with h5py.File(in_path + in_name_h5py, 'r') as f:
    params = dict(f['params'].attrs.items())
vbins = params['v_bins']

X = dill.load(open(dir_lhs + f'{z_string}_param{train_num}.p', 'rb')) # load normalized cosmological parameters from grab_models.py
X_test = dill.load(open(dir_lhs + f'{z_string}_param{test_num}.p', 'rb'))
X_vali = dill.load(open(dir_lhs + f'{z_string}_param{vali_num}.p', 'rb'))
meanX = X.mean(axis=0)
stdX = X.std(axis=0)
X_train = (X - meanX) / stdX
X_test = (X_test - meanX) / stdX
X_vali = (X_vali - meanX) / stdX
print(f'meanX = {meanX}')
print(f'stdX = {stdX}')
print(f'train: {X_train.shape}')

Y = dill.load(open(dir_lhs + f'{z_string}_model{train_num}.p', 'rb'))
Y_test = dill.load(open(dir_lhs + f'{z_string}_model{test_num}.p', 'rb'))
Y_vali = dill.load(open(dir_lhs + f'{z_string}_model{vali_num}.p', 'rb'))
meanY = Y.mean(axis=0)
stdY = Y.std(axis=0)
Y_train = (Y - meanY) / stdY
Y_test = (Y_test - meanY) / stdY
Y_vali = (Y_vali - meanY) / stdY
print(Y_vali.shape)

'''
Build custom haiku Module
'''
custom_forward = hk.without_apply_rng(hk.transform(_bnn_custom_forward_fn,apply_rng=True))
init_params = custom_forward.init(rng=next(hk.PRNGSequence(my_rng)), x=X_train)
n_samples = X_train.shape[0]
total_steps = n_epochs*n_samples + n_epochs

optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm),
                        optax.adamw(learning_rate=schedule_lr(lr,total_steps),weight_decay=decay)
                        )


'''
Training Loop + Visualization of params
'''
#params_grads_distribution(loss_fn,init_params,X_train,Y_train)

best_loss = np.inf
validation_loss = []
training_loss = []
early_stopping_counter = 0
pv = 100
#params = init_params
params = dict(
        # Haiku inits weights to trun. normal, with stddev ``1 / sqrt(fan_in)``.
        # Where ``fan_in`` is the number of incoming connection to the layer.
        mu=init_params,
        # Init to ~0.001 variance around default Haiku initialization.
        logvar=jax.tree_map(lambda x: -7 * jnp.ones_like(x), init_params),
    )
opt_state = optimizer.init(params)
IPython.embed()

if __name__ == "__main__":
    with trange(n_epochs) as t:
        for step in t:
            # optimizing loss by update function
            params, opt_state, batch_loss, grads = update(params, opt_state, X_train, Y_train, optimizer, my_rng, like_dict,vbins)

            # compute training & validation loss at the end of the epoch
            l = loss_fn(params, X_vali, Y_vali, my_rng, like_dict,vbins)
            training_loss.append(batch_loss)
            validation_loss.append(l)

            # update the progressbar
            t.set_postfix(loss=validation_loss[-1])

            # early stopping condition
            if l <= best_loss:
                best_loss = l
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            if early_stopping_counter >= pv:
                break
                
    best_params = params
    print(f'Reached max number of epochs in this batch. Validation loss ={best_loss}. Training loss ={batch_loss}')
    print(f'Model saved.')
    print(f'early_stopping_counter: {early_stopping_counter}')
    print(f'accuracy: {jnp.sqrt(jnp.mean(accuracy(params, X_test, Y_test, meanY, stdY, my_rng)**2))}')
    print(f'Test Loss: {loss_fn(params, X_test, Y_test, my_rng, like_dict,vbins)}')

    
    '''
    Plots
    '''
    preds = predict(best_params,X_train,my_rng)

    test_preds = predict(best_params,X_test,my_rng)
    test_loss = loss_fn(best_params, X_test, Y_test, my_rng, like_dict,vbins)
    test_R2 = r2_score(test_preds.squeeze(), Y_test)
    test_accuracy = (self.Y_test * self.stdY - test_preds * self.stdY) / (
                self.Y_test * self.stdY + self.meanY)  # relative error of test dataset
    self.RelativeError = test_accuracy

    plt.plot(range(len(validation_loss)), validation_loss, label=f'vali loss:{best_loss:.4f}')  # plot validation loss
    plt.plot(range(len(training_loss)), training_loss, label=f'train loss:{batch_loss: .4f}')  # plot training loss
    plt.legend()
    plt.savefig(os.path.join(dir_exp, f'epoch_loss_{out_tag}_{var_tag}.png'))

    # Fitting plots
    train_overplot(preds, X,Y,meanY,stdY, out_tag, var_tag)
    test_overplot(test_preds, Y_test, X_test, meanX, stdX, meanY, stdY, out_tag,
                  var_tag)

    # Accuracy + Results Plots
    plot_residue(self.RelativeError, out_tag, var_tag)
    plot_error_distribution(self.RelativeError, out_tag, var_tag)
    print(f'***Result Plots saved {dir_exp}***')  # imported from utils_plot
    '''
    Accuracy + Results
    '''
    delta = np.asarray(accuracy(best_params, X_test, Y_test, meanY, stdY, my_rng))
    print('Test R^2 Score: {}\n'.format(test_R2))  # R^2 score: ranging 0~1, 1 is good model

    '''
    Save best emulated parameter
    '''
    '''
    #small bin size
    if small_bin_bool==True:
        f = h5py.File(os.path.expanduser('~') + f'/igm_emulator/igm_emulator/emulator/best_params/z{redshift}_chi_bin59_savefile.hdf5', 'a')
    else:
        f = h5py.File(os.path.expanduser('~') + f'/igm_emulator/igm_emulator/emulator/best_params/z{redshift}_chi_savefile.hdf5', 'a')
    group1 = f.create_group('haiku_nn')
    group1.attrs['redshift'] = redshift
    group1.attrs['adamw_decay'] = decay
    group1.attrs['epochs'] = n_epochs
    group1.create_dataset('layers', data = output_size)
    group1.attrs['activation_function'] = f'{activation}'
    group1.attrs['learning_rate'] = lr
    group1.attrs['L2_lambda'] = l2

    group2 = f.create_group('data')
    group2.attrs['train_dir'] = dir_lhs + f'{z_string}_param{train_num}.p'
    group2.attrs['test_dir'] = dir_lhs + f'{z_string}_param{test_num}.p'
    group2.attrs['vali_dir'] = dir_lhs + f'{z_string}_param{vali_num}.p'
    group2.create_dataset('test_data', data = X_test)
    group2.create_dataset('train_data', data = X_train)
    group2.create_dataset('vali_data', data = X_vali)
    group2.create_dataset('meanX', data=meanX)
    group2.create_dataset('stdX', data=stdX)
    group2.create_dataset('meanY', data=meanY)
    group2.create_dataset('stdY', data=stdY)
    #IPython.embed()
    group3 = f.create_group('performance')
    group3.attrs['R2'] = test_R2
    group3.attrs['test_loss'] = test_loss
    group3.attrs['train_loss'] = batch_loss
    group3.attrs['vali_loss'] = best_loss
    group3.attrs['residuals_results'] = f'{jnp.mean(delta)*100}% +/- {jnp.std(delta) * 100}%'
    group3.create_dataset('residuals', data=delta)
    f.close()
    print("training directories and hyperparameters saved")

    dir = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/best_params'
    dir2 = '/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params'
    dill.dump(best_params, open(os.path.join(dir, f'{z_string}_chi_best_param{train_num}.p'), 'wb'))
    dill.dump(best_params, open(os.path.join(dir2, f'{z_string}_chi_best_param{train_num}.p'), 'wb'))
    print("trained parameter for smaller bins saved")
    '''