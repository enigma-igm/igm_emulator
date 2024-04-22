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
from jax.scipy.stats.multivariate_normal import logpdf
from functools import partial
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
import sys
import os
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator')
from utils_mlp import schedule_lr, loss_fn, accuracy, update, MyModuleCustom
from utils_plot import *
#sys.path.append(os.path.expanduser('~') + '/LAF_emulator/laf_emulator/emulators/flax_lite/trainers')
#from base import TrainerModule as LAFTrainerModule
import h5py
import IPython
config.update("jax_enable_x64", True)
dtype=jnp.float64

'''
Training Loop Module + Visualization of params
'''
class TrainerModule:

    def __init__(self,
                X_train: Any,
                Y_train: Any,
                X_test: Any,
                Y_test: Any,
                X_vali: Any,
                Y_vali: Any,
                layer_sizes: Sequence[int],
                x_scaler: Callable,
                y_scaler: Callable,
                activation: Callable[[jnp.ndarray], jnp.ndarray],
                dropout_rate: float,
                optimizer_hparams: Sequence[Any],
                loss_str: str,
                loss_weights: Sequence[Any],
                like_dict: dict,
                out_tag: str,
                bach_size = None,
                init_rng=42,
                n_epochs=1000,
                pv=100):

        super().__init__()

        #Set dataset and scaler functions
        self.X_train = X_train #transformed x
        self.Y_train = Y_train #transformed y
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_vali = X_vali
        self.Y_vali = Y_vali
        self.x_scaler = x_scaler #DiffStandardScaler class: scaler.transform(theta) and scaler.inverse_transform(x)
        self.y_scaler = y_scaler #DiffStandardScaler class: scaler.transform(theta) and scaler.inverse_transform(x)
        self.meanX = x_scaler.mean
        self.stdX = x_scaler.std
        self.meanY = y_scaler.mean
        self.stdY = y_scaler.std
        self.covar_nn = None
        self.err_nn = None

        #Set MLP parameters
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.optimizer_hparams = optimizer_hparams
        self.loss_str = loss_str
        self.l2_weight, self.c_loss, self.percent_loss = loss_weights
        self.like_dict = like_dict
        self.out_tag = out_tag
        self.var_tag =f'{self.loss_str}_l2_{self.l2_weight}_perc_{self.percent_loss}_activation_{activation.__name__}'
        self.init_rng = init_rng
        self.n_epochs = n_epochs
        self.pv = pv
        self.batch_size = bach_size
        def _custom_forward_fn(x):
            module = MyModuleCustom(output_size=self.layer_sizes, activation=self.activation,
                                    dropout_rate=self.dropout_rate)
            return module(x)
        self.custom_forward = hk.without_apply_rng(hk.transform(_custom_forward_fn))

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, params, X, Y):
        _loss_fn = jax.tree_util.Partial(loss_fn, like_dict=self.like_dict, custom_forward=self.custom_forward, l2=self.l2_weight, c_loss=self.c_loss, loss_str=self.loss_str, percent=self.percent_loss, scaler=self.y_scaler)
        return _loss_fn(params, X, Y)

    @partial(jit, static_argnums=(0,5))
    def train_step(self, params, opt_state, X, Y, optimizer):
        _update = jax.tree_util.Partial(update, like_dict=self.like_dict, custom_forward=self.custom_forward, l2=self.l2_weight, c_loss=self.c_loss, loss_str=self.loss_str, percent=self.percent_loss, scaler=self.y_scaler)
        return _update(params, opt_state, X, Y, optimizer)

    def create_batches(self, rstate, batch_size):

        # first we shuffle the data
        X, Y = shuffle(self.X_train, self.Y_train, random_state=rstate)
        n_batches = X.shape[0] // batch_size
        batches = []

        for i in np.arange(n_batches):
            single_batch = {
                'X': X[i * batch_size:(i + 1) * batch_size, :].reshape((batch_size, X.shape[1])),
                'Y': Y[i * batch_size:(i + 1) * batch_size, :].reshape((batch_size, Y.shape[1]))}
            batches.append(single_batch)

        return batches

    def train_loop(self, plot=True):
        '''
        Training loop for the neural network
        Parameters
        ----------
        plot

        Returns
        -------

        '''
        custom_forward = self.custom_forward
        params = custom_forward.init(rng=next(hk.PRNGSequence(jax.random.PRNGKey(self.init_rng))), x=self.X_train)

        n_samples = self.X_train.shape[0]
        if self.batch_size is not None:
            total_steps = self.n_epochs * (n_samples // self.batch_size)
        else:
            total_steps = self.n_epochs * n_samples + self.n_epochs

        max_grad_norm, lr, decay = self.optimizer_hparams
        optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm),
                                optax.adamw(learning_rate=schedule_lr(lr,total_steps),weight_decay=decay)
                                )

        opt_state = optimizer.init(params)

        early_stopping_counter = 0
        best_loss = np.inf
        validation_loss = []
        training_loss = []
        print(f'***Training Loop Start***')
        print(f'MLP info: {self.var_tag}')
        with trange(self.n_epochs) as t:
            for step in t:
                # optimizing loss by update function

                # go through each batch
                if self.batch_size is not None:
                    all_batches = self.create_batches(rstate=step, batch_size=self.batch_size)
                    for batch in all_batches:
                        params, opt_state, batch_loss, grads = self.train_step(params, opt_state, batch['X'], batch['Y'],
                                                                           optimizer)
                else:
                    params, opt_state, batch_loss, grads = self.train_step(params, opt_state, self.X_train, self.Y_train,
                                                                       optimizer)

                # compute training & validation loss at the end of the epoch
                l = self.loss_fn(params, self.X_vali, self.Y_vali)
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

        self.best_params = params

        #Validation Metrics
        vali_preds = custom_forward.apply(self.best_params, self.X_vali)
        self.best_chi_2_loss = -logpdf(x=vali_preds * self.stdY, mean=self.Y_vali * self.stdY, cov=self.like_dict['covariance'])
        print(f'Reached max number of epochs in this batch. Validation loss ={best_loss}. Training loss ={batch_loss}')
        print(f'early_stopping_counter: {early_stopping_counter}')
        print(f'Test Loss: {self.loss_fn(params, self.X_test, self.Y_test)}')

        #Test Metrics
        self.batch_loss = batch_loss
        test_preds = custom_forward.apply(self.best_params, self.X_test)
        test_accuracy = (self.Y_test*self.stdY-test_preds*self.stdY)/(self.Y_test*self.stdY+self.meanY) #relative error of test dataset
        self.RelativeError = test_accuracy
        self.test_chi_loss = ((test_preds - self.Y_test) * self.stdY) / jnp.sqrt(jnp.diagonal(self.like_dict['covariance']))
        print(f'Test accuracy: {jnp.sqrt(jnp.mean(jnp.square(test_accuracy)))}')

        self.test_loss = self.loss_fn(params, self.X_test, self.Y_test)
        self.vali_loss = self.loss_fn(params, self.X_vali, self.Y_vali)
        self.test_R2 = r2_score(test_preds.squeeze(), self.Y_test)
        print('Test R^2 Score: {}\n'.format(self.test_R2))  # R^2 score: ranging 0~1, 1 is good model
        preds = custom_forward.apply(self.best_params, self.X_train)

        if plot:
            #Prediction overplots: Training And Test
            plt.plot(range(len(validation_loss)), validation_loss, label=f'vali loss:{best_loss:.4f}')  # plot validation loss
            plt.plot(range(len(training_loss)), training_loss, label=f'train loss:{batch_loss: .4f}')  # plot training loss
            plt.legend()
            plt.savefig(os.path.join(dir_exp, f'epoch_loss_{self.out_tag}_{self.var_tag}.png'))

            #Fitting plots
            train_overplot(preds, self.X_train, self.Y_train, self.meanY, self.stdY, self.out_tag, self.var_tag)
            test_overplot(test_preds, self.Y_test, self.X_test,self.meanX,self.stdX,self.meanY,self.stdY, self.out_tag, self.var_tag)

            #Accuracy + Results Plots
            plot_residue(self.test_chi_loss,self.out_tag, self.var_tag)
            #bad_learned_plots(self.RelativeError,self.X_test,self.Y_test,test_preds,self.meanY,self.stdY, self.out_tag, self.var_tag)
            plot_error_distribution(self.RelativeError,self.out_tag,self.var_tag)
            print(f'***Result Plots saved {dir_exp}***') # imported from utils_plot

        return self.best_params, self.vali_loss

    def nn_error_propagation(self, theta_v, corr_v, save=False, err_vali_num=None):
        '''
        To propogate emulation error to the covariance matrix in inference

        Parameters
        ----------
        theta_v: in physical dimension
        corr_v

        Returns
        -------

        '''
        pred_v = self.y_scaler.inverse_transform(self.custom_forward.apply(self.best_params, self.x_scaler.transform(theta_v)))
        delta_v = corr_v - pred_v
        self.err_nn = delta_v.mean(axis=0)
        self.covar_nn = 1/(delta_v.shape[0]-1) * jnp.cov((delta_v-self.err_nn).T)

        if save:
            dir2 = '/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params'
            dill.dump(self.covar_nn, open(os.path.join(dir2, f'{self.out_tag}{err_vali_num}_{self.var_tag}_covar_nn.p'), 'wb'))
            dill.dump(self.err_nn, open(os.path.join(dir2, f'{self.out_tag}{err_vali_num}_{self.var_tag}_err_nn.p'), 'wb'))
            dill.dump(delta_v, open(os.path.join(dir2, f'{self.out_tag}{err_vali_num}_{self.var_tag}_delta_vali_nn.p'), 'wb'))
            dill.dump(self.like_dict['covariance'], open(os.path.join(dir2, f'{self.out_tag}_{self.var_tag}_covar_data.p'), 'wb'))

        return self.covar_nn, self.err_nn, delta_v

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
            group1.attrs['epochs'] = self.n_epochs
            group1.create_dataset('layers', data = self.layer_sizes)
            group1.attrs['activation_function'] = self.activation.__name__
            group1.attrs['learning_rate'] = lr
            group1.attrs['L2_weight'] = self.l2_weight
            group1.attrs['loss_fn'] = self.loss_str

            group2 = f.create_group('data')
            group2.create_dataset('test_param', data = self.X_test*self.stdX+self.meanX)
            group2.create_dataset('train_param', data = self.X_train*self.stdX+self.meanX)
            group2.create_dataset('vali_param', data = self.X_vali*self.stdX+self.meanX)
            group2.create_dataset('test_model', data = self.Y_test*self.stdY+self.meanY)
            group2.create_dataset('train_model', data = self.Y_train*self.stdY+self.meanY)
            group2.create_dataset('vali_model', data = self.Y_vali*self.stdY+self.meanY)
            group2.create_dataset('meanX', data=self.meanX)
            group2.create_dataset('stdX', data=self.stdX)
            group2.create_dataset('meanY', data=self.meanY)
            group2.create_dataset('stdY', data=self.stdY)
            #IPython.embed()
            group3 = f.create_group('performance')
            group3.attrs['R2'] = self.test_R2
            group3.attrs['test_loss'] = self.test_loss
            group3.attrs['train_loss'] = self.batch_loss
            group3.attrs['vali_loss'] = self.vali_loss
            group3.attrs['residuals_results'] = f'{jnp.mean(self.RelativeError)*100}% +/- {jnp.std(self.RelativeError) * 100}%'
            group3.create_dataset('residuals', data=self.RelativeError)
            f.close()
            print("training directories and hyperparameters saved")

            dir2 = '/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params'
            dill.dump(self.best_params, open(os.path.join(dir2, f'{self.out_tag}_{self.var_tag}_best_param.p'), 'wb'))
            dill.dump(self.covar_nn, open(os.path.join(dir2, f'{self.out_tag}_{self.var_tag}_covar_nn.p'), 'wb'))
            dill.dump(self.err_nn, open(os.path.join(dir2, f'{self.out_tag}_{self.var_tag}_err_nn.p'), 'wb'))
            print("trained parameters saved")

