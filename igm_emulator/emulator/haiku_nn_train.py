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

redshift = 5.4
num = 3
# get the appropriate string and pathlength for chosen redshift
zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
z_idx = np.argmin(np.abs(zs - redshift))
z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
z_string = z_strings[z_idx]

dir_lhs = '/home/zhenyujin/igm_emulator/igm_emulator/emulator/LHS/'
X = dill.load(open(dir_lhs + f'{z_string}_normparam{num}.p', 'rb'))
Y = dill.load(open(dir_lhs + f'{z_string}_model{num}.p', 'rb'))
Y = Y/((Y.max())-(Y.min()))

class haiku_nn:

    def FeedForward(x):
        mlp = hk.nets.MLP(output_sizes=self.layers)
        return mlp(x)

    model = hk.transform(FeedForward)

    def __init__(
            self,
            layer_size = (200,200,200,276),
            rng = jax.random.PRNGKey(42),
            epochs = 2000,
            learning_rate = 0.001,
            patience_values = 100,
            X_train: jnp.ndarray = [],
            Y_train: jnp.ndarray = []
    ):

        self.epochs = epochs
        self.lr = learning_rate
        self.pv = patience_values
        self.layers = layer_size

        self.x = X_train
        self.y = Y_train

        self.params = haiku_nn.model.init(rng, self.x)
        self.preds = haiku_nn.model.apply(self.params, rng, self.x)

    def MeanSquaredErrorLoss(params, x, y):
        compute_loss = jnp.mean((model.apply(params, rng, x) - y) ** 2)
        return compute_loss

    def __call__(self):
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(self.params)

        patience_values = 100
        loss = []
        best_loss = np.inf
        early_stopping_counter = 0

        with trange(self.epochs) as t:
            for i in t:
                l, grads = value_and_grad(self.MeanSquaredErrorLoss)(self.params, self.x, self.y)
                updates, opt_state = optimizer.update(grads, opt_state)
                self.params = optax.apply_updates(self.params, updates)

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
                if early_stopping_counter >= patience_values:
                    print('Loss = ' + str(best_loss))
                    print('Model saved.')
                    break

            print('Reached max number of epochs. Loss = ' + str(best_loss))
            print('Model saved.')

        preds = self.preds

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

train = haiku_nn(X_train = jnp.array(X, dtype=jnp.float32), Y_train = jnp.array(Y, dtype=jnp.float32))
train()