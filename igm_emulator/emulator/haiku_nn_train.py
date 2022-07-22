import jax
import numpy as np
from jax import numpy as jnp
import haiku as hk
import optax
from matplotlib import pyplot as plt
from tqdm import trange
import dill
import os
from jax import value_and_grad
'''
Constructing Training Process of Haiku Neural Network:
'''
class haiku_nn:
# -*- Define arguments for training -*-
    def __init__(
            self,
            layer_size = (206,206,206,276),     #hidden layers and final layer(size of 276 is the size of the output)
            rng = jax.random.PRNGKey(42),       #for generating initialized weights and biases
            epochs = 1000,                      #epoch time for training
            learning_rate = 0.001,              #rate of changing weight parameters when learning
            patience_values = 100,              #number of increasing loss gradient, prevent from overlearning
            X_train: jnp.ndarray = [],          #input tensor of shape [sampling_size, input_dimension(=3)]
            Y_train: jnp.ndarray = []           #output tensor of shape [sampling_size, output_dimension(=276)]
    ):
        self.rng = rng
        self.epochs = epochs
        self.lr = learning_rate
        self.pv = patience_values
        self.layers = layer_size
        self.x = X_train
        self.y = Y_train
        # -----------------------------------------------------------------------
        #use MLP module in Haiku to initialize parameter and calculate predictions
        def FeedForward(x):
            mlp = hk.nets.MLP(output_sizes=self.layers)
            return mlp(x)
        model = hk.transform(FeedForward)
        self.model = model
        self.params_init = self.model.init(rng, self.x)
        # -----------------------------------------------------------------------

# -*- Define Loss function to be uodated on -*-
    def MeanSquaredErrorLoss(self, params):
        compute_loss = jnp.mean((self.model.apply(params, None, self.x) - self.y) ** 2)
        return compute_loss

# -*- Learning by minimizing the loss function -*-
    def train(self):
        # Select gradient optimizer with Optax, here we choose "adam"
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(self.params_init)

        # Initial conditions for arguments during
        loss = []
        best_loss = np.inf
        early_stopping_counter = 0
        params = self.params_init

        with trange(self.epochs) as t:
            for i in t:
                l, grads = value_and_grad(self.MeanSquaredErrorLoss)(params)
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

        preds = self.model.apply(params, None, self.x)

        # Plot
        dir_exp = '/home/zhenyujin/igm_emulator/igm_emulator/emulator/EXP/'
        ax = np.arange(276)
        sample = 5
        fig, axs = plt.subplots(sample, 1)
        for i in range(1, sample + 1):
            axs[i - 1].plot(ax, preds[i - 1, :], label=f'pred_{i}', linewidth=5)
            axs[i - 1].plot(ax, Y[i - 1, :], label=f'real_{i}', linewidth=2)
        plt.legend()
        plt.savefig(os.path.join(dir_exp, f'{self.layers}_{num}.png'))
        plt.show()

        fig, axs = plt.subplots(1, 1)
        for i in range(5):
            axs.plot(ax, preds[i], label=f'pred {i}', c=f'C{i}', alpha=0.3)
            axs.plot(ax, Y[i], label=f'real {i}', c=f'C{i}', linestyle='--')
        # axs.plot(ax, y_mean, label='Y mean', c='k', alpha=0.2)
        plt.legend()
        plt.savefig(os.path.join(dir_exp, f'{self.layers}_overplot_{num}.png'))
        plt.show()

'''
Load Data
'''
redshift = 5.4 #choose redshift from
num = 3 #choose data number of LHS sampling

# get the appropriate string and pathlength for chosen redshift
zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
z_idx = np.argmin(np.abs(zs - redshift))
z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
z_string = z_strings[z_idx]
dir_lhs = '/home/zhenyujin/igm_emulator/igm_emulator/emulator/LHS/'

X = dill.load(open(dir_lhs + f'{z_string}_normparam{num}.p', 'rb'))
Y = dill.load(open(dir_lhs + f'{z_string}_model{num}.p', 'rb'))
Y = Y/((Y.max())-(Y.min())) #normalize outout for better convergence

'''
Train Data
'''
data3 = haiku_nn(X_train = jnp.array(X, dtype=jnp.float32), Y_train = jnp.array(Y, dtype=jnp.float32))
data3.train()