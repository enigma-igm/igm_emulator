import jax
import numpy as np
from jax import numpy as jnp
import haiku as hk
import optax
from matplotlib import pyplot as plt
from tqdm import trange
import dill
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
#print(Y[1,:],Y[2,:])
dir_exp = '/home/zhenyujin/igm_emulator/igm_emulator/emulator/EXP/'
def save(attributes,filename):
    # attributes= [n_hidden,layer_sizes,X,Y,loss,w,b]
    # save attributes to file
    dill.dump(attributes,open(os.path.join(dir_exp, f'{filename}{num}.p'),'wb'))

Y = Y/((Y.max())-(Y.min()))

X_train, Y_train =  jnp.array(X, dtype=jnp.float32),\
                    jnp.array(Y, dtype=jnp.float32),\


n_hidden = 3
r = np.random.randint(100, 400, 1)
layer_size = np.ndarray.tolist(np.append([r,r,r],276))
print(layer_size)
def FeedForward(x):
    mlp = hk.nets.MLP(output_sizes=layer_size)
    return mlp(x)
model = hk.transform(FeedForward)

rng = jax.random.PRNGKey(42) ## Reproducibility ## Initializes model with same weights each time.
params = model.init(rng, X_train)
epochs = 2000
learning_rate = 0.001
patience_values = 100
loss = []
best_loss= np.inf
early_stopping_counter = 0
'''
def MeanSquaredErrorLoss(weights, input_data, actual):
    preds = model.apply(weights, rng, input_data)
    preds = preds.squeeze()
    #print(preds.shape,actual.shape)
    return jnp.power(actual - preds, 2).mean()
'''
def MeanSquaredErrorLoss(params, x, y):
    compute_loss =  jnp.mean((model.apply(params, rng, x) - y) ** 2)
    return compute_loss


optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

with trange(epochs) as t:
                for i in t:
                    l, grads = value_and_grad(MeanSquaredErrorLoss)(params, X_train, Y_train)
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
                    #print (early_stopping_counter)
                    if early_stopping_counter >= patience_values:
                        attributes = [n_hidden, layer_size, X, Y, best_loss, params]
                        #save(attributes,'emu5_200to400nrs')
                        print('Loss = ' + str(best_loss))
                        print('Model saved.')
                        break
                attributes = [n_hidden, layer_size, X, Y, best_loss, params]
                #save(attributes, 'emu5_200to400nrs')
                print('Reached max number of epochs. Loss = ' + str(best_loss))
                print('Model saved.')

preds = model.apply(params, rng, X_train)
#print(preds[1,:]-Y[1,:],preds[2,:]-Y[2,:])

ax = np.arange(276)
sample=5
fig, axs = plt.subplots(sample,1)
for i in range (1,sample+1):
    axs[i-1].plot(ax,preds[i-1,:],label=f'pred_{i}',linewidth=5)
    axs[i-1].plot(ax,Y[i-1,:],label=f'real_{i}',linewidth=2)
plt.legend()
plt.savefig(os.path.join(dir_exp, f'{layer_size}_{num}.png'))
plt.show()

fig, axs = plt.subplots(1, 1)
for i in range(5):
    axs.plot(ax, preds[i], label=f'pred {i}', c=f'C{i}', alpha=0.3)
    axs.plot(ax, Y[i], label=f'real {i}', c=f'C{i}', linestyle='--')
# axs.plot(ax, y_mean, label='Y mean', c='k', alpha=0.2)
plt.legend()
plt.savefig(os.path.join(dir_exp, f'{layer_size}_overplot_{num}.png'))
plt.show()

