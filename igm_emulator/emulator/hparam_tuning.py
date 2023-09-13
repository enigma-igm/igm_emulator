import jax
import optuna
from optuna.samplers import TPESampler
from emulator_train import X_train, Y_train, X_test, Y_test, X_vali, Y_vali, meanX, stdX, meanY, stdY, TrainerModule, out_tag, like_dict

def objective(trial):
    layer_sizes_tune = trial.suggest_categorical('layer_sizes', [[100, 100, 100, 59], [100, 100, 59], [100, 59]])
    activation_tune = trial.suggest_categorical('activation', ['jax.nn.leaky_relu', 'jax.nn.relu', 'jax.nn.sigmoid', 'jax.nn.tanh'])
    dropout_rate_tune = trial.suggest_float('dropout_rate', 0, 0.1)
    max_grad_norm_tune = trial.suggest_float('max_grad_norm', 0, 0.5)
    lr_tune = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    decay_tune = trial.suggest_float('decay', 1e-4, 5e-3, log=True)
    l2_tune = trial.suggest_float('l2', 1e-5, 1e-3, log=True)
    n_epochs_tune = trial.suggest_categorical('n_epochs', [500, 1000, 2000])
    loss_str_tune = trial.suggest_categorical('loss_str', ['chi_one_covariance', 'mse', 'mse+fft'])
    trainer = TrainerModule(X_train, Y_train, X_test, Y_test, X_vali, Y_vali, meanX, stdX, meanY, stdY,
                            layer_sizes= layer_sizes_tune,
                            activation=eval(activation_tune),
                            dropout_rate=dropout_rate_tune,
                            optimizer_hparams=[max_grad_norm_tune, lr_tune, decay_tune],
                            loss_str=loss_str_tune,
                            l2_weight=l2_tune,
                            like_dict=like_dict,
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
