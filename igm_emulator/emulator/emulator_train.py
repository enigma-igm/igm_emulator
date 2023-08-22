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
from jax import jit
from functools import partial
from sklearn.metrics import r2_score
import sys
import os
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator')
from haiku_custom_forward import schedule_lr, loss_fn, accuracy, update, MyModuleCustom
from plotVis import *
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/scripts')
import h5py
import IPython

max_grad_norm = 0.1
n_epochs = 1000
lr = 1e-3
beta = 1e-3
decay = 5e-3
l2 =0.0001
print('***Training Start***')
'''
print(f'Small bin number: {small_bin_bool}')
print(f'Layers: {output_size}')
print(f'Activation: {activation.__name__}')
print(f'L2 regularization lambda: {l2}')
print(f'Loss function: {loss_str}')
'''
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

    def __init__(self,
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
                loss_str: str,
                l2_weight: float,
                like_dict: dict,
                accuracy_fn: Callable,
                out_tag: str,
                init_rng=42,
                n_epochs=1000,
                pv=100):

        super().__init__()
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
        self.loss_str = loss_str
        self.l2_weight = l2_weight
        self.like_dict = like_dict
        self.accuracy_fn = accuracy_fn
        self.out_tag = out_tag
        self.var_tag =f'{loss_str}_l2_{l2_weight}_activation_{activation.__name__}_layers_{layer_sizes}'
        self.init_rng = init_rng
        self.n_epochs = n_epochs
        self.pv = pv
        def _custom_forward_fn(x):
            module = MyModuleCustom(output_size=self.layer_sizes, activation=self.activation,
                                    dropout_rate=self.dropout_rate)
            return module(x)
        self.custom_forward = hk.without_apply_rng(hk.transform(_custom_forward_fn))

    #@partial(jit, static_argnums=(0,))
    def loss_fn(self):
        return jax.tree_util.Partial(loss_fn, like_dict=self.like_dict, custom_forward=self.custom_forward, l2=self.l2_weight, loss_str=self.loss_str)

    #@partial(jit, static_argnums=(0,))
    def update(self):
        return jax.tree_util.Partial(update, like_dict=self.like_dict, custom_forward=self.custom_forward, l2=self.l2_weight, loss_str=self.loss_str)

    def train_loop(self):

#if __name__ == '__main__':
        custom_forward = self.custom_forward
        params = custom_forward.init(rng=next(hk.PRNGSequence(jax.random.PRNGKey(self.init_rng))), x=self.X_train)
        preds = custom_forward.apply(params, self.X_train)

        n_samples = self.X_train.shape[0]
        total_steps = self.n_epochs*n_samples + self.n_epochs
        max_grad_norm, lr, decay = self.optimizer_hparams
        optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm),
                                optax.adamw(learning_rate=schedule_lr(lr,total_steps),weight_decay=decay)
                                )

        opt_state = optimizer.init(params)
        early_stopping_counter = 0
        best_loss = np.inf
        validation_loss = []
        training_loss = []
        print('***Training Loop Start***')
        with trange(self.n_epochs) as t:
            for step in t:
                # optimizing loss by update function
                params, opt_state, batch_loss, grads = self.update()(params, opt_state, self.X_train, self.Y_train, optimizer)
                #if step % 100 == 0:
                    #plot_params(params)

                # compute training & validation loss at the end of the epoch
                l = self.loss_fn()(params, self.X_vali, self.Y_vali)
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
                if early_stopping_counter >= self.pv:
                    break

        print(f'Reached max number of epochs in this batch. Validation loss ={best_loss}. Training loss ={batch_loss}')
        self.best_params = params
        print(f'early_stopping_counter: {early_stopping_counter}')
        print(f'accuracy: {jnp.sqrt(jnp.mean(self.accuracy_fn(params, self.X_test, self.Y_test, self.meanY, self.stdY,custom_forward)**2))}')
        print(f'Test Loss: {self.loss_fn(params, self.X_test, self.Y_test)}')
        plt.plot(range(len(validation_loss)), validation_loss, label=f'vali loss:{best_loss:.4f}')  # plot validation loss
        plt.plot(range(len(training_loss)), training_loss, label=f'train loss:{batch_loss: .4f}')  # plot training loss
        plt.legend()

        #Prediction overplots: Training And Test

        print(f'***Result Plots saved {dir_exp}***')

        self.batch_loss = batch_loss
        test_preds = custom_forward.apply(self.best_params, self.X_test)
        self.test_loss = self.loss_fn()(params, self.X_test, self.Y_test)
        self.test_R2 = r2_score(test_preds.squeeze(), self.Y_test)
        print('Test R^2 Score: {}\n'.format(self.test_R2))  # R^2 score: ranging 0~1, 1 is good model
        preds = custom_forward.apply(self.best_params, X_train)

        train_overplot(preds, self.X, self.Y, self.meanY, self.stdY, out_tag)
        test_overplot(test_preds, self.Y_test, self.X_test,self.meanX,self.stdX,self.meanY,self.stdY,self.out_tag)

        #Accuracy + Results

        self.RelativeError = np.asarray(self.accuracy_fn(self.best_params, self.X_test, self.Y_test, self.meanY, self.stdY,custom_forward))

        plot_residue(self.RelativeError,out_tag)
        bad_learned_plots(self.RelativeError,self.X_test,self.Y_test,test_preds,self.meanY,self.stdY,self.out_tag)
        plot_error_distribution(self.RelativeError,self.out_tag)
        plot_error_distribution(self.RelativeError,self.out_tag,log=True)
        self.best_loss = best_loss

        return self.best_params, self.best_loss

    def save_training_info(self, redshift):
            zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
            z_idx = np.argmin(np.abs(zs - redshift))
            z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
            z_string = z_strings[z_idx]
            max_grad_norm, lr, decay = self.optimizer_hparams
            #Save best emulated parameter

            print(f'***Saving training info & best parameters***')

            f = h5py.File(os.path.expanduser('~') + f'/igm_emulator/igm_emulator/emulator/best_params/{self.out_tag}_{self.var_tag}_savefile.hdf5', 'a')
            group1 = f.create_group('haiku_nn')
            group1.attrs['redshift'] = redshift
            group1.attrs['adamw_decay'] = decay
            group1.attrs['epochs'] = n_epochs
            group1.create_dataset('layers', data = self.layer_sizes)
            group1.attrs['activation_function'] = self.activation.__name__
            group1.attrs['learning_rate'] = lr
            group1.attrs['L2_weight'] = self.l2_weight
            group1.attrs['loss_fn'] = self.loss_str

            group2 = f.create_group('data')
            group2.attrs['train_dir'] = dir_lhs + f'{z_string}_param{train_num}.p'
            group2.attrs['test_dir'] = dir_lhs + f'{z_string}_param{test_num}.p'
            group2.attrs['vali_dir'] = dir_lhs + f'{z_string}_param{vali_num}.p'
            group2.create_dataset('test_data', data = self.X_test)
            group2.create_dataset('train_data', data = self.X_train)
            group2.create_dataset('vali_data', data = self.X_vali)
            group2.create_dataset('meanX', data=self.meanX)
            group2.create_dataset('stdX', data=self.stdX)
            group2.create_dataset('meanY', data=self.meanY)
            group2.create_dataset('stdY', data=self.stdY)
            #IPython.embed()
            group3 = f.create_group('performance')
            group3.attrs['R2'] = self.test_R2
            group3.attrs['test_loss'] = self.test_loss
            group3.attrs['train_loss'] = self.batch_loss
            group3.attrs['vali_loss'] = self.best_loss
            group3.attrs['residuals_results'] = f'{jnp.mean(self.RelativeError)*100}% +/- {jnp.std(self.RelativeError) * 100}%'
            group3.create_dataset('residuals', data=self.RelativeError)
            f.close()
            print("training directories and hyperparameters saved")

            dir = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/best_params'
            dir2 = '/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params'
            dill.dump(self.best_params, open(os.path.join(dir, f'{out_tag}_{var_tag}_best_param.p'), 'wb'))
            dill.dump(self.best_params, open(os.path.join(dir2, f'{out_tag}_{var_tag}_best_param.p'), 'wb'))
            print("trained parameters saved")

trainer = TrainerModule(X_train,Y_train,X_test,Y_test,X_vali,Y_vali,meanY,stdY,
                        layer_sizes=[100,100,100,59],
                        activation= jax.nn.leaky_relu,
                        dropout_rate=0.1,
                        optimizer_hparams=[max_grad_norm, lr, decay],
                        loss_str='mse',
                        l2_weight=l2,
                        like_dict=like_dict,
                        accuracy_fn=accuracy,
                        init_rng=42,
                        n_epochs=1000,
                        pv=100,
                        out_tag=out_tag)
IPython.embed()
#train_loop(X_train, Y_train, X_test, Y_test, X_vali, Y_vali, meanY, stdY, params,
            #optimizer, update, loss_fn, accuracy, like_dict)