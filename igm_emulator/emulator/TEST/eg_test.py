import dill
import numpy as np
import jax.numpy as jnp
import jax
import elegy as eg
import optax
import matplotlib.pyplot as plt
import haiku as hk

''' 
Load Train and Test Data
'''
redshift = 5.4 #choose redshift from
num = '6_1000' #choose data number of LHS sampling
# get the appropriate string and pathlength for chosen redshift
zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
z_idx = np.argmin(np.abs(zs - redshift))
z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
z_string = z_strings[z_idx]
dir_lhs = '/igm_emulator/emulator/LHS/'

X_train = dill.load(open(dir_lhs + f'{z_string}_normparam{num}.p', 'rb')) # load normalized cosmological parameters from grab_models.py
Y_train = dill.load(open(dir_lhs + f'{z_string}_model{num}.p', 'rb'))
Y_train = Y_train/((Y_train.max())-(Y_train.min())) #normalize corr functions for better convergence

X_test = dill.load(open(dir_lhs + f'{z_string}_normparam4.p', 'rb')) # load normalized cosmological parameters from grab_models.py
Y_test = dill.load(open(dir_lhs + f'{z_string}_model4.p', 'rb'))
Y_test = Y_test/((Y_test.max())-(Y_test.min()))


class MLP(eg.Module):
    def __init__(self, n1: int, n2: int):
        super().__init__()
        self.__call__()
        self.n1 = n1
        self.n2 = n2

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.float32) / 255.0
        x = eg.nn.Flatten()(x)
        # first layers
        x = eg.nn.Linear(self.n1)(x)
        x = jax.nn.relu(x)
        # first layers
        x = eg.nn.Linear(self.n2)(x)
        x = jax.nn.relu(x)
        # output layer
        x = eg.nn.Linear(10)(x)

        return x

model = eg.Model(
    module=MLP.init(X_train),
    loss=[
        eg.losses.Crossentropy(),
        eg.regularizers.L2(l=1e-4),
    ],
    metrics=eg.metrics.mean_absolute_percentage_error(),
    optimizer=optax.adam(1e-3),
)

history = model.fit(
    inputs=X_train,
    labels=Y_train,
    epochs=100,
    steps_per_epoch=200,
    batch_size=64,
    validation_data=(X_test, Y_test),
    shuffle=True,
    callbacks=[eg.callbacks.ModelCheckpoint("models/high-level", save_best_only=True)],
)


def plot_history(history):
    n_plots = len(history.history.keys()) // 2
    plt.figure(figsize=(14, 24))

    for i, key in enumerate(list(history.history.keys())[:n_plots]):
        metric = history.history[key]
        val_metric = history.history[f"val_{key}"]

        plt.subplot(n_plots, 1, i + 1)
        plt.plot(metric, label=f"Training {key}")
        plt.plot(val_metric, label=f"Validation {key}")
        plt.legend(loc="lower right")
        plt.ylabel(key)
        plt.title(f"Training and Validation {key}")
    plt.show()


plot_history(history)