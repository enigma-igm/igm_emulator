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
from haiku_custom_forward import _custom_forward_fn, schedule_lr, loss_fn, accuracy, update, output_size, activation, l2
from plotVis import *
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/scripts')
from pytree_h5py import save, load
import h5py
import IPython

max_grad_norm = 0.1
n_epochs = 1000
lr = 1e-3
decay = 5e-3
print(f'Layers: {output_size}')
print(f'Activation: {activation}')
print(f'L2 regularization lambda: {l2}')
config.update("jax_enable_x64", True)
dtype=jnp.float64

'''
Load Train and Test Data
'''
redshift = 5.4 #choose redshift from
train_num = '_training_768_bin59'
test_num = '_test_89_bin59'
vali_num = '_vali_358_bin59'
# get the appropriate string and pathlength for chosen redshift
zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
z_idx = np.argmin(np.abs(zs - redshift))
z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
z_string = z_strings[z_idx]
dir_lhs = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/GRID/'


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
custom_forward = hk.without_apply_rng(hk.transform(_custom_forward_fn))
init_params = custom_forward.init(rng=42, x=X_train)
preds = custom_forward.apply(params=init_params, x=X_train)
n_samples = X_train.shape[0]
total_steps = n_epochs*n_samples + n_epochs

optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm),
                        optax.adamw(learning_rate=schedule_lr(lr,total_steps),weight_decay=decay)
                        )
opt_state = optimizer.init(init_params)
#train_overplot(preds,X,Y,meanY,stdY)

'''
Training Loop + Visualization of params
'''
#params_grads_distribution(loss_fn,init_params,X_train,Y_train)

best_loss = np.inf
validation_loss = []
training_loss = []
early_stopping_counter = 0
pv = 100
params = init_params

if __name__ == "__main__":
    with trange(n_epochs) as t:
        for step in t:
            # optimizing loss by update function
            params, opt_state, batch_loss, grads = update(params, opt_state, X_train, Y_train, optimizer)

            #if step % 100 == 0:
                #plot_params(params)

            # compute training & validation loss at the end of the epoch
            l = loss_fn(params, X_vali, Y_vali)
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

    print(f'Reached max number of epochs in this batch. Validation loss ={best_loss}. Training loss ={batch_loss}')
    best_params = params
    print(f'Model saved.')
    print(f'early_stopping_counter: {early_stopping_counter}')
    print(f'accuracy: {jnp.mean(accuracy(params, X_test, Y_test, meanY, stdY))}')
    print(f'Test Loss: {loss_fn(params, X_test, Y_test)}')
    plt.plot(range(len(validation_loss)), validation_loss, label=f'vali loss:{best_loss:.4f}')  # plot validation loss
    plt.plot(range(len(training_loss)), training_loss, label=f'train loss:{batch_loss: .4f}')  # plot training loss
    plt.legend()
    '''
    Prediction overplots: Training And Test
    '''
    preds = custom_forward.apply(params=best_params, x=X_train)
    train_overplot(preds,X,Y,meanY,stdY)

    test_preds = custom_forward.apply(params, X_test)
    test_loss = loss_fn(params, X_test, Y_test)
    test_R2 = r2_score(test_preds.squeeze(), Y_test)

    test_overplot(test_preds, Y_test, X_test,meanX,stdX,meanY,stdY)
    '''
    Accuracy + Results
    '''
    delta = np.asarray(accuracy(best_params, X_test, Y_test, meanY, stdY))
    print('Test R^2 Score: {}\n'.format(test_R2))  # R^2 score: ranging 0~1, 1 is good model
    print(f'accuracy: {jnp.mean(delta)*100}')

    plot_residue(delta)
    bad_learned_plots(delta,X_test,Y_test,test_preds,meanY,stdY)
    plot_error_distribution(delta)

    '''
    Save best emulated parameter
    '''

    f = h5py.File(os.path.expanduser('~') + f'/igm_emulator/igm_emulator/emulator/best_params/z{redshift}_nn_bin59_savefile.hdf5', 'w')
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
    save(os.path.expanduser('~') + f'/igm_emulator/igm_emulator/emulator/best_params/z{redshift}_nn_bin59_savefile.hdf5', best_params)
    save(f'/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_paramss/z{redshift}_nn_bin59_savefile.hdf5', best_params)
    #IPython.embed()
    dir = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/best_params'
    dir2 = '/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params'
    dill.dump(best_params, open(os.path.join(dir, f'{z_string}_best_param{train_num}.p'), 'wb'))
    dill.dump(best_params, open(os.path.join(dir2, f'{z_string}_best_param{train_num}.p'), 'wb'))
    print("trained parameter for smaller bins saved")
    IPython.embed()
