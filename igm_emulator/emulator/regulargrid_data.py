from igm_emulator.scripts.grab_models import param_transform
import dill
import os
import numpy as np
from matplotlib import pyplot as plt

# redshift to get models for -- can make this an input to this script if desired
redshift = 5.4
n_samples = 972
# get the appropriate string and pathlength for chosen redshift
zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
z_idx = np.argmin(np.abs(zs - redshift))
z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
z_string = z_strings[z_idx]
n_paths = np.array([17, 16, 16, 15, 15, 15, 14])
n_path = n_paths[z_idx]

# read in the parameter grid at given z
param_in_path = '/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/'
param_dict = dill.load(open(param_in_path + f'{z_string}_params.p', 'rb'))
# get the path to the autocorrelation function results from the simulations
in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'

fobs = param_dict['fobs']  # average observed flux <F> ~ Gamma_HI -9
log_T0s = param_dict['log_T0s']  # log(T_0) from temperature - density relation -15
T0s = np.exp(log_T0s)
gammas = param_dict['gammas']  # gamma from temperature - density relation -9
#print(fobs, T0s, gammas)


final_samples = np.empty([n_samples, 3])
x = np.linspace(0,1,9)
y = np.linspace(0,1,12)
z = np.linspace(0,1,9)
xg, yg, zg = np.meshgrid(x, y, z)
print (f'xg: {xg}')
print (f'yg: {yg}')
print (f'zg: {zg}')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xg, yg, zg)
# convert the output of lhs (between 0 and 1 for each parameter) to our model grid
xg_trans = param_transform(xg, fobs[0], fobs[-1]).flatten()

sample_params = np.array([param_transform(xg, fobs[0], fobs[-1]).flatten(),param_transform(yg, T0s[0], T0s[-1]).flatten(),param_transform(yg, gammas[0], gammas[-1]).flatten()])
sample_params = sample_params.T
print(sample_params.shape)


for sample_idx, sample in enumerate(sample_params):

    # determine the closest model to each lhs sample
    fobs_idx = np.argmin(np.abs(fobs - sample[0]))
    T0_idx = np.argmin(np.abs(T0s - sample[1]))
    gamma_idx = np.argmin(np.abs(gammas - sample[2]))

    # save the closest model parameters for each lhs sample
    final_samples[sample_idx, 0] = fobs[fobs_idx]
    final_samples[sample_idx, 1] = T0s[T0_idx]
    final_samples[sample_idx, 2] = gammas[gamma_idx]

    # get the corresponding model autocorrelation for each parameter location
    like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{gamma_idx}_SNR0_F{fobs_idx}_ncovar_500000_P{n_path}_set_bins_4.p'
    like_dict = dill.load(open(in_path + like_name, 'rb'))
    model_autocorrelation = like_dict['mean_data']
    if sample_idx == 0:
        models = np.empty([n_samples, len(model_autocorrelation)])
    models[sample_idx] = model_autocorrelation

print(final_samples.shape)
dir = '/home/zhenyujin/igm_emulator/igm_emulator/emulator/GRID'
num = '7_972'
#dill.dump(final_samples,open(os.path.join(dir, f'{z_string}_param{num}.p'),'wb'))
#dill.dump(models,open(os.path.join(dir, f'{z_string}_model{num}.p'),'wb'))

H = final_samples
# H= norm(loc=0, scale=1).ppf(lhd)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(H[:, 0], H[:, 1], H[:, 2], c=H[:, 2], cmap='viridis', linewidth=0.5)
ax.set_xlabel(r'$<F>$')
ax.set_ylabel(r'$T_0$')
ax.set_zlabel(r'$\gamma$')
plt.savefig("params.png")
plt.show()

