import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import dill
from scipy.interpolate import interpn, RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator



def gaussian(x, mu, sigma):
    """ Return the normalized Gaussian with standard deviation sigma. """
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / sigma / np.sqrt(2 * np.pi)

if __name__ == '__main__':

    model_in_path = '/mnt/quasar2/mawolfson/correlation_funct/models/'
    temp_models = np.loadtxt(model_in_path+'temp_models', skiprows=1)
    fred_models_1 = np.loadtxt(model_in_path+'fred_thermal_1', skiprows=0)

    fred_z = fred_models_1[:, 0]
    fred_t0_1 = fred_models_1[:, 1]
    fred_g_1 = fred_models_1[:, 2]


    redshifts = [5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]
    z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
    fred_f_1 = [0.0801, 0.0591, 0.0447, 0.0256, 0.0172, 0.0114, 0.0089]
    in_path_start = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final_135/'

    out_path = f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/all_z/'
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # define which dict to open
    n_path =  20
    R_value = 30000.
    skewers_use = 2000
    n_covar = 500000
    # bin_label = '_set_bins_2'
    bin_label = '_set_bins_3'
    n_temp = 15
    n_gamma = 9
    n_f = 9
    n_inference = 4000*4 #sample numbers (nwalkers * nsteps)
    n_mcmc = 3500
    n_walkers = 16
    n_skip = 500

    true_temp_idx = int(np.floor(n_temp / 2.))
    true_gamma_idx = int(np.floor(n_gamma / 2.))
    true_fobs_idx = int(np.floor(n_f / 2.))

    in_path_read = f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/hmc_results/central_models/'

    '''
    Central Mean model -- Emulator
    '''
    out_file_tag = 'mean'

    in_name_inference = f'{z_strings[0]}_{z_strings[-1]}_F{true_fobs_idx}_T0{true_temp_idx}_G{true_gamma_idx}_central_model_{out_file_tag}.hdf5'
    with h5py.File(in_path_read + in_name_inference, 'r') as f:
        samples_temp = np.array(f['samples_temp'])
        samples_gamma = np.array(f['samples_gamma'])
        samples_f = np.array(f['samples_f'])
        n_plot_rows = np.array(f.attrs['n_plot_rows'])
        reweight_ngp = False

    bins_temp = np.linspace(350, 19400, 40)
    pdf_hists_temp = np.empty([n_plot_rows, len(redshifts), len(bins_temp)-1])
    cdf_hists_temp = np.empty([n_plot_rows, len(redshifts), len(bins_temp)-1])

    bins_gamma = np.linspace(.382, 2.232, 40)
    pdf_hists_gamma = np.empty([n_plot_rows, len(redshifts), len(bins_gamma)-1])
    cdf_hists_gamma = np.empty([n_plot_rows, len(redshifts), len(bins_gamma)-1])

    bins_f = np.linspace(.0005, 0.0747, 40)
    pdf_hists_f = np.empty([n_plot_rows, len(redshifts), len(bins_f) - 1])
    cdf_hists_f = np.empty([n_plot_rows, len(redshifts), len(bins_f) - 1])

    max_temps = np.empty([len(redshifts)])
    min_temps = np.empty([len(redshifts)])
    max_gammas = np.empty([len(redshifts)])
    min_gammas = np.empty([len(redshifts)])
    max_f = np.empty([len(redshifts)])
    min_f = np.empty([len(redshifts)])

    for redshift_idx in range(len(redshifts)):
        '''
        Read params at given z
        '''
        param_file_name = f'correlation_temp_fluct_skewers_{skewers_use}_R_{int(R_value)}_nf_{n_f}_dict{bin_label}.hdf5'
        with h5py.File(in_path_start + f'{z_strings[redshift_idx]}/' + param_file_name, 'r') as f:
            params = dict(f['params'].attrs.items())

        R_value = params['R']
        sig_noise_ratios = params['SNRs']
        fobs = params['average_observed_flux']
        v_bins = params['v_bins']
        temps = 10 ** params['logT_0']
        gammas = params['gamma']

        max_temps[redshift_idx] = np.max(temps)
        min_temps[redshift_idx] = np.min(temps)
        max_gammas[redshift_idx] = np.max(gammas)
        min_gammas[redshift_idx] = np.min(gammas)
        max_f[redshift_idx] = np.max(fobs)
        min_f[redshift_idx] = np.min(fobs)

        print(f'z = {redshifts[redshift_idx]}')

        #Read mocks of temp samples at given z
        samples_temp_z = samples_temp[:, redshift_idx, :]
        samples_gamma_z = samples_gamma[:, redshift_idx, :]
        samples_f_z = samples_f[:, redshift_idx, :]


        # make one big histogram
        for mock_idx in range(n_plot_rows):

            hist_temp, bin_edges_temp = np.histogram(
                    samples_temp_z[mock_idx, :], bins=bins_temp,  density=True
                )
            hist_gamma, bin_edges_gamma = np.histogram(
                    samples_gamma_z[mock_idx, :], bins=bins_gamma, density=True
                )

            hist_f, bin_edges_f = np.histogram(
                        samples_f_z[mock_idx, :], bins=bins_f, density=True
                    )

            pdf_hists_temp[mock_idx, redshift_idx, :] = hist_temp
            mids_temp = (bin_edges_temp[:-1] + bin_edges_temp[1:])/2.
            cdf_hists_temp[mock_idx, redshift_idx, :] = np.cumsum(
                pdf_hists_temp[mock_idx, redshift_idx, :]
            )*np.diff(bin_edges_temp)[0]
            print(cdf_hists_temp[mock_idx, redshift_idx, -1])

            pdf_hists_gamma[mock_idx, redshift_idx, :] = hist_gamma
            mids_gamma = (bin_edges_gamma[:-1] + bin_edges_gamma[1:])/2.
            cdf_hists_gamma[mock_idx, redshift_idx, :] = np.cumsum(
                pdf_hists_gamma[mock_idx, redshift_idx, :]
            )*np.diff(bin_edges_gamma)[0]
            print(cdf_hists_gamma[mock_idx, redshift_idx, -1])

            pdf_hists_f[mock_idx, redshift_idx, :] = hist_f
            mids_f = (bin_edges_f[:-1] + bin_edges_f[1:])/2.
            cdf_hists_f[mock_idx, redshift_idx, :] = np.cumsum(
                    pdf_hists_f[mock_idx, redshift_idx, :]
                )*np.diff(bin_edges_f)[0]
            print(cdf_hists_f[mock_idx, redshift_idx, -1])

    print(f'overall max fob: {max_f}')
    print(f'overall min fob: {min_f}')

    print(f'overall max temp: {max_temps}')
    print(f'overall min temp: {min_temps}')


    print(f'overall max gamma: {max_gammas}')
    print(f'overall min gamma: {min_gammas}')


    x_size = 3.5
    dpi_value = 200

    plt_params = {'legend.fontsize': 7,
                  'legend.frameon': False,
                  'axes.labelsize': 8,
                  'axes.titlesize': 8,
                  'figure.titlesize': 8,
                  'xtick.labelsize': 7,
                  'ytick.labelsize': 7,
                  'lines.linewidth': .7,
                  'lines.markersize': 2.3,
                  'lines.markeredgewidth': .7,
                  'errorbar.capsize': 3,
                  'font.family': 'serif',
                  # 'text.usetex': True,
                  'xtick.minor.visible': True,
                  }
    plt.rcParams.update(plt_params)

    print('plotting')

    # add the model line

    redshifts = np.array(redshifts)
    limit_error = 2.

    # make the plot

    mfp_model_fig = plt.figure(figsize=(x_size*1.5, x_size*2.4*.55/2*1.5), constrained_layout=True,
                               dpi=dpi_value,
                               )
    mfp_model_fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
    height_ratios = n_plot_rows* [7, 7, 3]
    grid = mfp_model_fig.add_gridspec(
        nrows=n_plot_rows*3, ncols=1, height_ratios=height_ratios,  # width_ratios=[1, 1],
    )


    width_temp = 300
    width_gamma = .02
    width_f = .0002

    for mock_plot_idx in range(n_plot_rows):
        temp_axis = mfp_model_fig.add_subplot(grid[3*mock_plot_idx])
        gamma_axis = mfp_model_fig.add_subplot(grid[3*mock_plot_idx + 1])
        f_axis = mfp_model_fig.add_subplot(grid[3 * mock_plot_idx + 2])

        for redshift_idx_2 in range(len(redshifts)):
            mask_nonzero_temp = (pdf_hists_temp[mock_plot_idx, redshift_idx_2, :] > 0.000002)
            temp_axis.plot(
                redshifts[redshift_idx_2]+width_temp*pdf_hists_temp[mock_plot_idx, redshift_idx_2, mask_nonzero_temp],
                mids_temp[mask_nonzero_temp],
                color='k'
            )
            temp_axis.plot(
                redshifts[redshift_idx_2]-width_temp*pdf_hists_temp[mock_plot_idx, redshift_idx_2, mask_nonzero_temp],
                mids_temp[mask_nonzero_temp],
                color='k'
            )

            mask_1sig_temp = (cdf_hists_temp[mock_plot_idx, redshift_idx_2, :] > 0.16) & (cdf_hists_temp[mock_plot_idx, redshift_idx_2, :] < 0.84)
            mask_2sig_temp = (cdf_hists_temp[mock_plot_idx, redshift_idx_2, :] > 0.025) & (cdf_hists_temp[mock_plot_idx, redshift_idx_2, :] < 0.975)

            mask_nonzero_gamma = (pdf_hists_gamma[mock_plot_idx, redshift_idx_2, :] > 0.0005)
            gamma_axis.plot(
                redshifts[redshift_idx_2]+width_gamma*pdf_hists_gamma[mock_plot_idx, redshift_idx_2, mask_nonzero_gamma],
                mids_gamma[mask_nonzero_gamma],
                color='k'
            )
            gamma_axis.plot(
                redshifts[redshift_idx_2]-width_gamma*pdf_hists_gamma[mock_plot_idx, redshift_idx_2, mask_nonzero_gamma],
                mids_gamma[mask_nonzero_gamma],
                color='k'
            )

            mask_1sig_gamma = (cdf_hists_gamma[mock_plot_idx, redshift_idx_2, :] > 0.16) & (cdf_hists_gamma[mock_plot_idx, redshift_idx_2, :] < 0.84)
            mask_2sig_gamma = (cdf_hists_gamma[mock_plot_idx, redshift_idx_2, :] > 0.025) & (cdf_hists_gamma[mock_plot_idx, redshift_idx_2, :] < 0.975)

            mask_nonzero_f = (pdf_hists_f[mock_plot_idx, redshift_idx_2, :] > 0.00000001)
            f_axis.plot(
                redshifts[redshift_idx_2]+width_f*pdf_hists_f[mock_plot_idx, redshift_idx_2, mask_nonzero_f],
                mids_f[mask_nonzero_f],
                color='k'
            )
            f_axis.plot(
                redshifts[redshift_idx_2]-width_f*pdf_hists_f[mock_plot_idx, redshift_idx_2, mask_nonzero_f],
                mids_f[mask_nonzero_f],
                color='k'
            )

            mask_1sig_f = (cdf_hists_f[mock_plot_idx, redshift_idx_2, :] > 0.16) & (cdf_hists_f[mock_plot_idx, redshift_idx_2, :] < 0.84)
            mask_2sig_f = (cdf_hists_f[mock_plot_idx, redshift_idx_2, :] > 0.025) & (cdf_hists_f[mock_plot_idx, redshift_idx_2, :] < 0.975)


            if redshift_idx_2 == 0:
                temp_axis.fill_betweenx(
                    mids_temp[mask_2sig_temp],
                    redshifts[redshift_idx_2]+ width_temp * pdf_hists_temp[mock_plot_idx, redshift_idx_2, mask_2sig_temp],
                    redshifts[redshift_idx_2]- width_temp * pdf_hists_temp[mock_plot_idx, redshift_idx_2, mask_2sig_temp],
                    color='lightcoral' if out_file_tag == 'mean' else 'skyblue',
                    label=r'mean models 95% region' if out_file_tag == 'mean' else r'mock data 95% region',
                )
                temp_axis.fill_betweenx(
                    mids_temp[mask_1sig_temp],
                    redshifts[redshift_idx_2]+ width_temp * pdf_hists_temp[mock_plot_idx, redshift_idx_2, mask_1sig_temp],
                    redshifts[redshift_idx_2]- width_temp * pdf_hists_temp[mock_plot_idx, redshift_idx_2, mask_1sig_temp],
                    color='orangered' if out_file_tag == 'mean' else 'dodgerblue',
                    label=r'mean models 68% region' if out_file_tag == 'mean' else r'mock data 68% region',
                )

                gamma_axis.fill_betweenx(
                    mids_gamma[mask_2sig_gamma],
                    redshifts[redshift_idx_2]+ width_gamma * pdf_hists_gamma[mock_plot_idx, redshift_idx_2, mask_2sig_gamma],
                    redshifts[redshift_idx_2]- width_gamma * pdf_hists_gamma[mock_plot_idx, redshift_idx_2, mask_2sig_gamma],
                    color='lightcoral' if out_file_tag == 'mean' else 'skyblue',
                    label=r'mean models 95% region' if out_file_tag == 'mean' else r'mock data 95% region',
                )
                gamma_axis.fill_betweenx(
                    mids_gamma[mask_1sig_gamma],
                    redshifts[redshift_idx_2]+ width_gamma * pdf_hists_gamma[mock_plot_idx, redshift_idx_2, mask_1sig_gamma],
                    redshifts[redshift_idx_2]- width_gamma * pdf_hists_gamma[mock_plot_idx, redshift_idx_2, mask_1sig_gamma],
                    color='orangered' if out_file_tag == 'mean' else 'dodgerblue',
                    label=r'mean models 68% region' if out_file_tag == 'mean' else r'mock data 68% region',
                )
                f_axis.fill_betweenx(
                    mids_f[mask_2sig_f],
                    redshifts[redshift_idx_2]+ width_f * pdf_hists_f[mock_plot_idx, redshift_idx_2, mask_2sig_f],
                    redshifts[redshift_idx_2]- width_f * pdf_hists_f[mock_plot_idx, redshift_idx_2, mask_2sig_f],
                    color='lightcoral' if out_file_tag == 'mean' else 'skyblue',
                    label=r'mean models 95% region' if out_file_tag == 'mean' else r'mock data 95% region',
                )
                f_axis.fill_betweenx(
                    mids_f[mask_1sig_f],
                    redshifts[redshift_idx_2] + width_f * pdf_hists_f[mock_plot_idx, redshift_idx_2, mask_1sig_f],
                    redshifts[redshift_idx_2] - width_f * pdf_hists_f[mock_plot_idx, redshift_idx_2, mask_1sig_f],
                    color='orangered' if out_file_tag == 'mean' else 'dodgerblue',
                    label=r'mean models 68% region' if out_file_tag == 'mean' else r'mock data 68% region',
                )


            else:
                temp_axis.fill_betweenx(
                    mids_temp[mask_2sig_temp],
                    redshifts[redshift_idx_2]+ width_temp * pdf_hists_temp[mock_plot_idx, redshift_idx_2, mask_2sig_temp],
                    redshifts[redshift_idx_2]- width_temp * pdf_hists_temp[mock_plot_idx, redshift_idx_2, mask_2sig_temp],
                    color='lightcoral' if out_file_tag == 'mean' else 'skyblue',
                )
                temp_axis.fill_betweenx(
                    mids_temp[mask_1sig_temp],
                    redshifts[redshift_idx_2]+ width_temp * pdf_hists_temp[mock_plot_idx, redshift_idx_2, mask_1sig_temp],
                    redshifts[redshift_idx_2]- width_temp * pdf_hists_temp[mock_plot_idx, redshift_idx_2, mask_1sig_temp],
                    color='orangered' if out_file_tag == 'mean' else 'dodgerblue',
                )

                gamma_axis.fill_betweenx(
                    mids_gamma[mask_2sig_gamma],
                    redshifts[redshift_idx_2]+ width_gamma * pdf_hists_gamma[mock_plot_idx, redshift_idx_2, mask_2sig_gamma],
                    redshifts[redshift_idx_2]- width_gamma * pdf_hists_gamma[mock_plot_idx, redshift_idx_2, mask_2sig_gamma],
                    color='lightcoral' if out_file_tag == 'mean' else 'skyblue',
                )
                gamma_axis.fill_betweenx(
                    mids_gamma[mask_1sig_gamma],
                    redshifts[redshift_idx_2]+ width_gamma * pdf_hists_gamma[mock_plot_idx, redshift_idx_2, mask_1sig_gamma],
                    redshifts[redshift_idx_2]- width_gamma * pdf_hists_gamma[mock_plot_idx, redshift_idx_2, mask_1sig_gamma],
                    color='orangered' if out_file_tag == 'mean' else 'dodgerblue',
                )

                f_axis.fill_betweenx(
                    mids_f[mask_2sig_f],
                    redshifts[redshift_idx_2]+ width_f * pdf_hists_f[mock_plot_idx, redshift_idx_2, mask_2sig_f],
                    redshifts[redshift_idx_2]- width_f * pdf_hists_f[mock_plot_idx, redshift_idx_2, mask_2sig_f],
                    color='lightcoral' if out_file_tag == 'mean' else 'skyblue',
                )
                f_axis.fill_betweenx(
                    mids_f[mask_1sig_f],
                    redshifts[redshift_idx_2] + width_f * pdf_hists_f[mock_plot_idx, redshift_idx_2, mask_1sig_f],
                    redshifts[redshift_idx_2] - width_f * pdf_hists_f[mock_plot_idx, redshift_idx_2, mask_1sig_f],
                    color='orangered' if out_file_tag == 'mean' else 'dodgerblue',
                )

        temp_axis.plot(fred_z, fred_t0_1, linestyle='--', color='k', label='model')
        gamma_axis.plot(fred_z, fred_g_1, linestyle='--', color='k', label='model')
        f_axis.plot(fred_z,  np.interp(fred_z, redshifts, fred_f_1), linestyle='--', color='k', label='model')

        temp_axis.text(5.315, 16000., f'mean models' if out_file_tag == 'mean' else f'mock {mock_plot_idx + 1}', {'color': 'k', 'fontsize': 7}, )

        temp_axis.set_ylabel(r'$T_{{0}}$ (K)')
        gamma_axis.set_ylabel(r'$\gamma$')
        f_axis.set_ylabel(r'$\langle F \rangle$')
        #f_axis.set_yscale('log')

        gamma_axis.set_xlim([5.301, 6.099])
        temp_axis.set_xlim([5.301, 6.099])
        f_axis.set_xlim([5.301, 6.099])

        if mock_plot_idx == 0:
            temp_axis.legend(frameon=True, bbox_to_anchor=(1.1, 1.3), loc='upper right')
        else:
            temp_axis.set_title(' ')
            gamma_axis.set_title(' ')

        if mock_plot_idx == n_plot_rows - 1:
            f_axis.tick_params(
                axis='x',
                which='both',
                bottom=True,
                labelbottom=True
            )
            f_axis.set_xlabel('Redshift')

        temp_axis.set_ylim([min_temps[-1], max_temps[0]])
        gamma_axis.set_ylim([min_gammas[0], max_gammas[-1]])
        f_axis.set_ylim([0.0007, 0.085])

        temp_axis.tick_params(
            axis='x',
            which='both',
            bottom=False,
            labelbottom=False
        )
        gamma_axis.tick_params(
            axis='x',
            which='both',
            bottom=False,
            labelbottom=False
        )
    '''
    Random 5 mocks
    '''
    #save_name = f'thermal_state_measurements_sigma_shaded_violin_F{true_fobs_idx}_T{true_temp_idx}_G{true_gamma_idx}_R_{int(R_value)}{bin_label}_plot_mocks_{n_plot_rows}'
    '''
    Molly's 2 mocks
    '''
    save_name = f'thermal_state_measurements_sigma_shaded_violin_3_rows_F{true_fobs_idx}_T{true_temp_idx}_G{true_gamma_idx}_R_{int(R_value)}{bin_label}_plot_{out_file_tag}'

    mfp_model_fig.savefig(out_path + f'{save_name}.pdf')
    mfp_model_fig.savefig(out_path + f'{save_name}.png')