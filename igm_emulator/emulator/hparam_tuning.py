import jax
import optuna
from optuna.samplers import TPESampler
from emulator_run import X_train, Y_train, X_test, Y_test, X_vali, Y_vali, meanX, stdX, meanY, stdY, out_tag, like_dict
from emulator_train import TrainerModule
def objective(trial):
    layer_sizes_tune = trial.suggest_categorical('layer_sizes', [(100, 100, 100, 59), (100, 100, 59), (100, 59)])
    activation_tune = trial.suggest_categorical('activation', ['jax.nn.leaky_relu', 'jax.nn.relu', 'jax.nn.sigmoid', 'jax.nn.tanh'])
    dropout_rate_tune = trial.suggest_categorical('dropout_rate', [None, 0.05, 0.1])
    max_grad_norm_tune = trial.suggest_float('max_grad_norm', 0, 0.5, step=0.1)
    lr_tune = trial.suggest_float('lr', 1e-5,1e-3, log=True)
    decay_tune = trial.suggest_float('decay', 1e-4, 5e-3, log=True)
    l2_tune = trial.suggest_categorical('l2', [0, 1e-5, 1e-4, 1e-3])
    c_loss_tune = trial.suggest_float('c_loss', 1e-3, 1, log=True)
    percent_loss_tune = trial.suggest_categorical('percent', [True, False])
    n_epochs_tune = trial.suggest_categorical('n_epochs', [500, 1000, 2000])
    loss_str_tune = trial.suggest_categorical('loss_str', ['chi_one_covariance', 'mse', 'mse+fft', 'huber', 'mae'])
    trainer = TrainerModule(X_train, Y_train, X_test, Y_test, X_vali, Y_vali, meanX, stdX, meanY, stdY,
                            layer_sizes= layer_sizes_tune,
                            activation=eval(activation_tune),
                            dropout_rate=dropout_rate_tune,
                            optimizer_hparams=[max_grad_norm_tune, lr_tune, decay_tune],
                            loss_str=loss_str_tune,
                            loss_weights=[l2_tune,c_loss_tune,percent_loss_tune],
                            like_dict=like_dict,
                            small_bin_bool=True,
                            init_rng=42,
                            n_epochs=n_epochs_tune,
                            pv=100,
                            out_tag=out_tag)

    best_vali_loss = trainer.train_loop(False)[1]
    del trainer
    return best_vali_loss

print('*** Running the hyperparameter tuning ***')

# create the study
number_of_trials = 50
sampler = TPESampler(seed=10)  # 10
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=number_of_trials, gc_after_trial=True)

trial = study.best_trial
print(f'\nBest Validation Loss: {trial.value}')
print(f'Best Params:')
for key, value in trial.params.items():
    print(f'-> {key}: {value}')
print()
