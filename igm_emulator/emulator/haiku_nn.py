import jax
import numpy as np
from jax import numpy as jnp
import haiku as hk
import optax
from matplotlib import pyplot as plt
from tqdm import trange
import os
from jax import value_and_grad
from sklearn.metrics import r2_score

'''
Constructing Haiku Neural Network:
    haiku_nn(X_train, Y_train): create emulator with chosen properties on train data
    haiku_nn.train(): training loop for emulator to learn on traian data
    haiku_nn.test(X_test, Y_test): evaluate performance of trained emulator on new LHS test data, including loss, R^2 score, and relative error plot
'''
class haiku_nn:

# -*- Define arguments for training -*-
    def __init__(
            self,
            layer_size = (100,100,100,100,276),     #hidden layers and final layer(size of 276 is the size of the output)
            rng = jax.random.PRNGKey(42),       #for generating initialized weights and biases
            epochs = 1000,                      #epoch time for training
            learning_rate = 0.0003,             #rate of changing weight parameters when learning
            patience_values = 100,              #number of increasing loss gradient, prevent from overlearning
            X_train: jnp.ndarray = [],          #input tensor of shape [sampling_size, input_dimension(=3)]
            Y_train: jnp.ndarray = [],           #output tensor of shape [sampling_size, output_dimension(=276)]
            comment = 'test4_mse_norm'
    ):
        self.rng = rng
        self.epochs = epochs
        self.lr = learning_rate
        self.pv = patience_values
        self.layers = layer_size
        self.x = X_train
        self.y = Y_train
        self.comment = comment
        # -----------------------------------------------------------------------
        #use MLP module in Haiku to initialize parameter and calculate predictions
        def FeedForward(x):
            mlp = hk.nets.MLP(output_sizes=self.layers)
            return mlp(x)
        model = hk.transform(FeedForward)

        self.model = model
        self.params_init = self.model.init(rng, self.x)
        # -----------------------------------------------------------------------


# -*- Define Loss functions to be updated on -*-
    def MeanSquaredErrorLoss(self, params, X, Y):
        compute_loss = jnp.mean((self.model.apply(params, None, X) - Y) ** 2)
        return compute_loss

    def RelativeError(self, params, X, Y):
        delta = jnp.mean(jnp.abs(self.model.apply(params, None, X) - Y)/Y)
        return delta

# -*- Learning process by minimizing the loss function -*-
    def train(self):
        # Select loss optimizer with Optax, here we choose "adam"
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(self.params_init)

        # Initial conditions for arguments during training
        loss = []
        best_loss = np.inf
        early_stopping_counter = 0
        params = self.params_init #initializing weight parameters

        # Training loop
        with trange(self.epochs) as t:
            for i in t:
                # optimizing loss by gradient descent
                l, grads = value_and_grad(self.MeanSquaredErrorLoss)(params, self.x, self.y)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)

                # compute validation loss at the end of the epoch
                loss.append(l)

                # update the progressbar
                t.set_postfix(loss=loss[-1])

                # early stopping condition
                if l <= best_loss:
                    best_loss = l
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                # print (early_stopping_counter)
                if early_stopping_counter >= self.pv:
                    print('Loss = ' + str(best_loss))
                    print('Model saved.')
                    break

            print('Reached max number of epochs. Loss = ' + str(best_loss))
            print(f'Model of size {self.layers} saved.')
        # Final trained parameters and resulting prediction
        self.params = params
        preds = self.model.apply(params, None, self.x)

        # Plot partial predited corrolation functions and
        ax = np.arange(276)                                                 # arbitrary even spaced x-axis (will be converted to velocityZ)
        sample = 5                                                          # number of functions plotted
        fig, axs = plt.subplots(1, 1)
        corr_idx = np.random.randint(0, 100, sample)                        # randomly select correlation functions to compare
        for i in range(sample):
            axs.plot(ax, preds[corr_idx[i]], label=f'Preds {i}:' r'$<F>$='f'{self.x[corr_idx[i],0]:.2f},'
                                                            r'$T_0$='f'{self.x[corr_idx[i],1]:.2f},'
                                                            r'$\gamma$='f'{self.x[corr_idx[i],2]:.2f}', c=f'C{i}', alpha=0.3)
            axs.plot(ax, self.y[corr_idx[i]], label=f'Real {i}', c=f'C{i}', linestyle='--')
        # axs.plot(ax, y_mean, label='Y mean', c='k', alpha=0.2)
        plt.xlabel(r'Will be changed to Velocity/ $km s^{-1}$')
        plt.ylabel('Correlation function')
        plt.title(f'Train Loss = {best_loss}')
        plt.legend()
        dir_exp = '/home/zhenyujin/igm_emulator/igm_emulator/emulator/EXP/'  # plot saving directory
        plt.savefig(os.path.join(dir_exp, f'{self.layers}_overplot{self.comment}.png'))
        plt.show()


# -*- Test emulator with new LHS data -*-
    def test(self, X_test, Y_test):
        test_preds = self.model.apply(self.params, None, X_test)
        test_loss = self.MeanSquaredErrorLoss(self.params, X_test, Y_test)
        test_R2 = r2_score(test_preds.squeeze(), Y_test)

        # Print performance of emulator on test data
        print(self.comment)
        print("Test MSE Loss: {}\n".format(test_loss)) # Loss
        print('Test R^2 Score: {}\n'.format(test_R2))  # R^2 score: ranging 0~1, 1 is good model

        # Plot relative error of all test correlation functions
        delta = (Y_test - test_preds)/Y_test*100
        ax = np.arange(276)
        for i in range(delta.shape[0]):
            #plt.ylim(-4,)
            plt.plot(ax,delta[i,:], linewidth=0.5)
        plt.xlabel(r'Will be changed to Velocity/ $km s^{-1}$')
        plt.ylabel('% error on Correlation function')
        plt.title(f'Test Loss = {test_loss:.6f}, R^2 Score = {test_R2:.4f}')
        dir_exp = '/home/zhenyujin/igm_emulator/igm_emulator/emulator/EXP/'  # plot saving directory
        plt.savefig(os.path.join(dir_exp, f'{self.layers}_test%error{self.comment}.png'))
        plt.show()

