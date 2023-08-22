import dill
import os
import numpy as np
import haiku as hk
import jax.numpy as jnp
import jax
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
import optax
from tqdm import trange
from jax.config import config
from sklearn.metrics import r2_score
import sys
import os
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator')
from haiku_custom_forward import schedule_lr, loss_fn, accuracy, update, output_size, activation, l2, small_bin_bool, var_tag, loss_str, MyModuleCustom
from plotVis import *
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/scripts')
import h5py
import IPython

max_grad_norm = 0.1
n_epochs = 1000
lr = 1e-3
beta = 1e-3
decay = 5e-3
print('***Training Start***')
print(f'Small bin number: {small_bin_bool}')
print(f'Layers: {output_size}')
print(f'Activation: {activation.__name__}')
print(f'L2 regularization lambda: {l2}')
print(f'Loss function: {loss_str}')
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
    out_tag = f'{z_string}{train_num}'
else:
    train_num = '_training_768'
    test_num = '_test_89'
    vali_num = '_vali_358'
    n_path = 17
    n_covar = 500000
    bin_label = '_set_bins_4'
    in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'
    out_tag = f'{z_string}{train_num}_bin276'

T0_idx = 8 #0-14
g_idx = 4 #0-8
f_idx = 4 #0-8
   
like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{g_idx}_SNR0_F{f_idx}_ncovar_{n_covar}_P{n_path}{bin_label}.p'
like_dict = dill.load(open(in_path + like_name, 'rb'))


X = dill.load(open(dir_lhs + f'{z_string}_param{train_num}.p', 'rb')) # load normalized cosmological parameters from grab_models.py
X_test = dill.load(open(dir_lhs + f'{z_string}_param{test_num}.p', 'rb'))
X_vali = dill.load(open(dir_lhs + f'{z_string}_param{vali_num}.p', 'rb'))
meanX = X.mean(axis=0)
stdX = X.std(axis=0)
X_train = (X - meanX) / stdX
X_test = (X_test - meanX) / stdX
X_vali = (X_vali - meanX) / stdX


Y = dill.load(open(dir_lhs + f'{z_string}_model{train_num}.p', 'rb'))
Y_test = dill.load(open(dir_lhs + f'{z_string}_model{test_num}.p', 'rb'))
Y_vali = dill.load(open(dir_lhs + f'{z_string}_model{vali_num}.p', 'rb'))
meanY = Y.mean(axis=0)
stdY = Y.std(axis=0)
Y_train = (Y - meanY) / stdY
Y_test = (Y_test - meanY) / stdY
Y_vali = (Y_vali - meanY) / stdY

print('***Data Loaded***')
print(f'Train datasize: {X_train.shape[0]}; Test datasize: {X_test.shape[0]}; Validation datasize: {X_vali.shape[0]}')

'''
Training Loop + Visualization of params
'''
class TrainerModule:

    def __int__(self,
                X_train: Any,
                Y_train: Any,
                X_test: Any,
                Y_test: Any,
                X_vali: Any,
                Y_vali: Any,
                meanY: Any,
                stdY: Any,
                layer_sizes: Sequence[int],
                activation: Callable[[jnp.ndarray], jnp.ndarray],
                dropout_rate: float,
                optimizer_hparams: Sequence[float],
                update: Callable,
                loss_str: str,
                l2_weight: float,
                accuracy_fn: Callable,
                schedule_lr: Callable,
                like_dict: dict,
                init_rng=42,
                n_epochs=1000,
                pv=100,
                save_training_info=False):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_vali = X_vali
        self.Y_vali = Y_vali
        self.meanY = meanY
        self.stdY = stdY
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.optimizer_hparams = optimizer_hparams
        self.update = update
        self.loss_str = loss_str
        self.l2_weight = l2_weight
        self.accuracy_fn = accuracy_fn
        self.schedule_lr = schedule_lr
        self.like_dict = like_dict
        self.init_rng = init_rng
        self.n_epochs = n_epochs
        self.pv = pv
        self.save_training_info = save_training_info

'''
    @jax.jit
    def train_loop(self):

#if __name__ == '__main__':
   
    def _custom_forward_fn(x):
        module = MyModuleCustom(output_size=layer_sizes, activation = activation, dropout_rate=dropout_rate)
        return module(x)
    custom_forward = hk.without_apply_rng(hk.transform(_custom_forward_fn))

    params = custom_forward.init(rng=next(hk.PRNGSequence(jax.random.PRNGKey(init_rng))), x=X_train)
    preds = custom_forward.apply(params, X_train)

    n_samples = X_train.shape[0]
    total_steps = n_epochs*n_samples + n_epochs
    max_grad_norm, lr, decay = optimizer_hparams
    optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm),
                            optax.adamw(learning_rate=schedule_lr(lr,total_steps),weight_decay=decay)
                            )

    opt_state = optimizer.init(params)
    early_stopping_counter = 0
    best_loss = np.inf
    validation_loss = []
    training_loss = []
    print('***Training Loop Start***')
    with trange(n_epochs) as t:
        for step in t:
            # optimizing loss by update function
            params, opt_state, batch_loss, grads = update(params, opt_state, X_train, Y_train, optimizer,like_dict, custom_forward, l2_weight)
            #if step % 100 == 0:
                #plot_params(params)

            # compute training & validation loss at the end of the epoch
            l = loss_fn(params, X_vali, Y_vali,like_dict, custom_forward=custom_forward, l2=l2_weight)
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
    print(f'early_stopping_counter: {early_stopping_counter}')
    print(f'accuracy: {jnp.sqrt(jnp.mean(accuracy_fn(params, X_test, Y_test, meanY, stdY,custom_forward)**2))}')
    print(f'Test Loss: {loss_fn(params, X_test, Y_test, like_dict, custom_forward, l2_weight)}')
    plt.plot(range(len(validation_loss)), validation_loss, label=f'vali loss:{best_loss:.4f}')  # plot validation loss
    plt.plot(range(len(training_loss)), training_loss, label=f'train loss:{batch_loss: .4f}')  # plot training loss
    plt.legend()


    #Prediction overplots: Training And Test

    print(f'***Result Plots saved {dir_exp}***')

    test_preds = custom_forward.apply(best_params, X_test)
    test_loss = loss_fn(params, X_test, Y_test,like_dict, custom_forward, l2_weight)
    test_R2 = r2_score(test_preds.squeeze(), Y_test)
    print('Test R^2 Score: {}\n'.format(test_R2))  # R^2 score: ranging 0~1, 1 is good model
    preds = custom_forward.apply(best_params, X_train)

    train_overplot(preds, X, Y, meanY, stdY,out_tag)
    test_overplot(test_preds, Y_test, X_test,meanX,stdX,meanY,stdY,out_tag)

    #Accuracy + Results

    delta = np.asarray(accuracy_fn(best_params, X_test, Y_test, meanY, stdY,custom_forward))

    plot_residue(delta,out_tag)
    bad_learned_plots(delta,X_test,Y_test,test_preds,meanY,stdY,out_tag)
    plot_error_distribution(delta,out_tag)
    plot_error_distribution(delta,out_tag,log=True)

    if save_training_info:
        zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
        z_idx = np.argmin(np.abs(zs - redshift))
        z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
        z_string = z_strings[z_idx]

        #Save best emulated parameter

        print(f'***Saving training info & best parameters***')

        f = h5py.File(os.path.expanduser('~') + f'/igm_emulator/igm_emulator/emulator/best_params/{out_tag}_{var_tag}_savefile.hdf5', 'a')
        group1 = f.create_group('haiku_nn')
        group1.attrs['redshift'] = redshift
        group1.attrs['adamw_decay'] = decay
        group1.attrs['epochs'] = n_epochs
        group1.create_dataset('layers', data = output_size)
        group1.attrs['activation_function'] = f'{activation}'
        group1.attrs['learning_rate'] = lr
        group1.attrs['L2_weight'] = l2_weight
        group1.attrs['loss_fn'] = loss_str

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
        dill.dump(best_params, open(os.path.join(dir, f'{out_tag}_{var_tag}_best_param.p'), 'wb'))
        dill.dump(best_params, open(os.path.join(dir2, f'{out_tag}_{var_tag}_best_param.p'), 'wb'))
        print("trained parameters saved")
    return best_params, best_loss
'''
IPython.embed()
#train_loop(X_train, Y_train, X_test, Y_test, X_vali, Y_vali, meanY, stdY, params,
            #optimizer, update, loss_fn, accuracy, like_dict)