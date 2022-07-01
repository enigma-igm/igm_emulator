
import os
import numpy as np
from matplotlib import pyplot as plt
import dill
from astropy.table import Table
from igm_emulator.emulator.utils import lhs


def param_transform(x, mins, maxs):
    """
    Function to transform output of latin hyper into parameter space
    Parameters
    ----------
    x
    mins
    maxs

    Returns
    -------

    """
    y = (maxs - mins) * x + mins
    return y


# Routine to randomly subsample a set of correlation function simulations
if __name__ == '__main__':
    # redshift to get models for -- can make this an input to this script if desired
    redshift = 5.4

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

    fobs = param_dict['fobs']  # average observed flux <F> ~ Gamma_HI
    log_T0s = param_dict['log_T0s']  # log(T_0) from temperature - density relation
    gammas = param_dict['gammas']  # gamma from temperature - density relation

    # get the path to the autocorrelation function results from the simulations
    in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'

    seed = 12345
    # random number generator
    rng = np.random.default_rng(seed)

    # get n_samples (15) latin hypercube sampling samples
    n_samples = 15
    samples = lhs(3, samples=n_samples)

    final_samples = np.empty([n_samples, 3])
    for sample_idx, sample in enumerate(samples):
        # convert the output of lhs (between 0 and 1 for each parameter) to our model grid
        sample_params = param_transform(sample,
                                        np.array([fobs[0], log_T0s[0], gammas[0]]),
                                        np.array([fobs[-1], log_T0s[-1], gammas[-1]]))

        # determine the closest model to each lhs sample
        fobs_idx = np.argmin(np.abs(fobs - sample_params[0]))
        log_T0_idx = np.argmin(np.abs(log_T0s - sample_params[1]))
        gamma_idx = np.argmin(np.abs(gammas - sample_params[2]))

        # save the closest model parameters for each lhs sample
        final_samples[sample_idx, 0] = fobs[fobs_idx]
        final_samples[sample_idx, 1] = log_T0s[log_T0_idx]
        final_samples[sample_idx, 2] = gammas[gamma_idx]

        # get the corresponding model autocorrelation for each parameter location
        like_name = f'likelihood_dicts_R_30000_nf_9_T{log_T0_idx}_G{gamma_idx}_SNR0_F{fobs_idx}_ncovar_500000_P{n_path}_set_bins_4.p'
        like_dict = dill.load(open(in_path + like_name, 'rb'))
        model_autocorrelation = like_dict['mean_data']
        if sample_idx == 0:
            models = np.empty([n_samples, len(model_autocorrelation)])
        models[sample_idx] = model_autocorrelation

    # now you have the parameters (final samples) and the corresponding auto-correlation values (models)
    # for each n_samples (initially written for 15) of the latin hypercube sampling results
    print(final_samples)
    H = final_samples
    # H= norm(loc=0, scale=1).ppf(lhd)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(H[:, 0], H[:, 1], H[:, 2], c=H[:, 2], cmap='viridis', linewidth=0.5)
    ax.set_xlabel(r'$<F>$')
    ax.set_ylabel(r'$log(T_0)$')
    ax.set_zlabel(r'$\gamma$')
    plt.savefig("params.png")
    plt.show()