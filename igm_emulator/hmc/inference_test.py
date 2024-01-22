import sys
import os
from hmc_nn_inference import NN_HMC_X
import igm_emulator as emu
import dill
import numpy as np
import IPython
import jax
import jax.random as random
from sklearn.metrics import mean_squared_error,r2_score
from scipy.spatial.distance import minkowski
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as pe
from tabulate import tabulate
from igm_emulator.emulator.emulator_apply import trainer
from igm_emulator.emulator.hparam_tuning import small_bin_bool
from mcmc_ngp_inference import HMC_NGP
sys.path.append('/home/zhenyujin/qso_fitting/')
import h5py
from qso_fitting.analysis.inf_test import run_inference_test, compute_importance_weights, C_ge, inference_test_plot
import corner
import h5py
from progressbar import ProgressBar

x_size = 3.5
dpi_value = 200
plt_params = {'legend.fontsize': 7,
              'legend.frameon': False,
              'axes.labelsize': 8,
              'axes.titlesize': 8,
              'figure.titlesize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'lines.linewidth': 1,
              'lines.markersize': 2,
              'errorbar.capsize': 3,
              'font.family': 'serif',
              # 'text.usetex': True,
              'xtick.minor.visible': True,
              }
plt.rcParams.update(plt_params)
class INFERENCE_TEST():
    def __init__(self, redshift,
                 model_emulator_bool, # True: emulator model; False: NGP model
                 gaussian_bool,  #True: Gaussian sampling around mean; False: Forward mocks sampling
                 ngp_bool, #True: NGP for theta points in inferece; False: theta points on priors
                 emu_test_bool=False, #True: perfect inference test with emulator mocks; False: inference test with mocks
                 true_log_prob_on_prior_bool=True, # If the true LogP is NGP or on prior: Boundary problem
                 n_inference=100, n_params=3,
                 out_path = '/mnt/quasar2/zhenyujin/igm_emulator/hmc/hmc_results/',
                 key_sample=36,
                 key_hmc=642
    ):


        # get the appropriate string and pathlength for chosen redshift
        zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
        z_idx = np.argmin(np.abs(zs - redshift))
        z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
        self.z_string = z_strings[z_idx]
        self.model_emulator_bool = model_emulator_bool
        self.gaussian_bool = gaussian_bool
        self.ngp_bool = ngp_bool
        self.n_inference = n_inference
        self.n_params = n_params
        self.emu_test_bool = emu_test_bool
        self.true_log_prob_on_prior = true_log_prob_on_prior_bool
        self.out_path = out_path
        self.key_sample = key_sample
        self.key_hmc = key_hmc

        if small_bin_bool == True:
            self.n_path = 20  # 17->20
            self.n_covar = 500000
            self.bin_label = '_set_bins_3'
            self.in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final_135/{self.z_string}/'
        else:
            self.n_path = 17
            self.n_covar = 500000
            self.bin_label = '_set_bins_4'
            self.in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{self.z_string}/final_135/'


        # load model from emulator_apply.py
        in_path_hdf5 = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/best_params/'
        self.var_tag = trainer.var_tag
        self.out_tag = trainer.out_tag
        self.best_params = dill.load(open(in_path_hdf5 + f'{self.out_tag}_{self.var_tag}_best_param.p', 'rb')) #changed to optuna tuned best param
        if self.model_emulator_bool==False:
            self.var_tag += '_NGP_model'
        '''
        Load Parameter Grid
        '''
        in_name_h5py = f'correlation_temp_fluct_skewers_2000_R_30000_nf_9_dict{self.bin_label}.hdf5'
        with h5py.File(self.in_path + in_name_h5py, 'r') as f:
            params = dict(f['params'].attrs.items())
        self.fobs = params['average_observed_flux']
        self.R_value = params['R']
        self.v_bins = params['v_bins']
        self.T0s = 10. ** params['logT_0']
        self.gammas = params['gamma']
        self.n_f = len(self.fobs)
        self.n_temps = len(self.T0s)
        self.n_gammas = len(self.gammas)
        self.noise_idx = 0

        T0_idx = 8
        g_idx = 4
        f_idx = 4
        like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{g_idx}_SNR0_F{f_idx}_ncovar_{self.n_covar}_P{self.n_path}{self.bin_label}.p'
        self.like_dict = dill.load(open(self.in_path + like_name, 'rb'))


    def mocks_sampling(self):
        if self.gaussian_bool == False:
            self.ngp_bool = True
            self.note = 'forward_mocks_ngp_emulator_prior_diff_covar'
        else:
            if self.ngp_bool == True and self.emu_test_bool == False:
                self.note = 'gaussian_ngp_mocks_prior_diff_covar'
            elif self.ngp_bool == True and self.emu_test_bool == True:
                self.note = 'gaussian_ngp_emulator_test_prior_diff_covar'
            else:
                self.note = 'gaussian_emulator_test_prior_diff_covar'
        print('Sampling parameters from priors')

        # get n_inference sampled parameters
        true_theta_sampled = np.empty([self.n_inference, self.n_params])
        rng = random.PRNGKey(self.key_sample)

        rng, init_rng = random.split(rng)
        true_temp = random.uniform(init_rng, (self.n_inference,), minval=self.T0s[0], maxval=self.T0s[-1])

        rng, init_rng = random.split(rng)
        true_gamma = random.uniform(init_rng, (self.n_inference,), minval=self.gammas[0], maxval=self.gammas[-1])

        rng, init_rng = random.split(rng)
        true_fob = random.uniform(init_rng, (self.n_inference,), minval=self.fobs[0], maxval=self.fobs[-1])

        true_theta_sampled[:, 1] = true_temp
        true_theta_sampled[:, 2] = true_gamma
        true_theta_sampled[:, 0] = true_fob


        # get n_inference mock correlation functions
        mock_corr = np.empty([self.n_inference, len(self.v_bins)])
        model_corr = np.empty([self.n_inference, len(self.v_bins)])
        mock_covar = np.empty([self.n_inference, len(self.v_bins), len(self.v_bins)])
        true_theta = np.empty([self.n_inference, self.n_params])
        pbar = ProgressBar()

        for mock_idx in pbar(range(self.n_inference)):

            closest_temp_idx = np.argmin(np.abs(self.T0s - true_theta_sampled[mock_idx, 1]))
            closest_gamma_idx = np.argmin(np.abs(self.gammas - true_theta_sampled[mock_idx, 2]))
            closest_fobs_idx = np.argmin(np.abs(self.fobs - true_theta_sampled[mock_idx, 0]))
            model_name = f'likelihood_dicts_R_30000_nf_9_T{closest_temp_idx}_G{closest_gamma_idx}_SNR0_F{closest_fobs_idx}_ncovar_{self.n_covar}_P{self.n_path}{self.bin_label}.p'
            model_dict = dill.load(open(self.in_path + model_name, 'rb'))
            model_corr[mock_idx, :] = model_dict['mean_data']
            cov = model_dict['covariance']
            if self.gaussian_bool:
                if self.ngp_bool:
                    true_theta[mock_idx, :] = [self.fobs[closest_fobs_idx], self.T0s[closest_temp_idx], self.gammas[closest_gamma_idx]]
                    mean = model_corr[mock_idx, :]
                    if self.emu_test_bool:
                        mean = emu.nn_emulator(self.best_params, true_theta[mock_idx, :])
                elif self.ngp_bool==False and self.emu_test_bool==True:
                    true_theta[mock_idx, :] = true_theta_sampled[mock_idx,
                                              :]  # for emulator, true_theta = true_theta_sampled: off-grid
                    mean = emu.nn_emulator(self.best_params, true_theta[mock_idx, :])
                    cov = self.like_dict['covariance']
                else:
                    raise Exception("For off-grid inference test, must use emulator.")
                #split rng!
                rng, init_rng = random.split(rng)

                mock_corr[mock_idx, :] = random.multivariate_normal(init_rng, mean, cov)
                mock_covar[mock_idx, :, :] = cov

            else:
                if self.ngp_bool:
                    true_theta[mock_idx, :] = [self.fobs[closest_fobs_idx], self.T0s[closest_temp_idx], self.gammas[closest_gamma_idx]]
                else:
                    true_theta[mock_idx, :] = true_theta_sampled[mock_idx,:]

                mock_name = f'mocks_R_{int(self.R_value)}_nf_{self.n_f}_T{closest_temp_idx}_G{closest_gamma_idx}_SNR{self.noise_idx}_F{closest_fobs_idx}_P{self.n_path}{self.bin_label}.p'
                mocks = dill.load(open(self.in_path + mock_name, 'rb'))
                #split rng!
                rng, init_rng = random.split(rng)

                mock_corr[mock_idx, :] = random.choice(key=init_rng, a=mocks, shape=(1,), replace=False)
                mock_covar[mock_idx, :, :] = cov

        # save get n_inference sampled parameters and mock correlation functions
        dill.dump(mock_corr, open(self.out_path + f'{self.note}_mock_corr_inference{self.n_inference}_{self.var_tag}.p', 'wb'))
        dill.dump(model_corr, open(self.out_path + f'{self.note}_model_corr_inference{self.n_inference}_{self.var_tag}.p', 'wb'))
        dill.dump(true_theta, open(self.out_path + f'{self.note}_theta_inference{self.n_inference}_{self.var_tag}.p', 'wb'))
        dill.dump(true_theta_sampled, open(self.out_path + f'{self.note}_theta_sampled_inference{self.n_inference}_{self.var_tag}.p', 'wb'))
        dill.dump(mock_covar, open(self.out_path + f'{self.note}_covar_inference{self.n_inference}_{self.var_tag}.p', 'wb'))

        self.mock_corr = mock_corr
        self.mock_covar = mock_covar
        self.model_corr = model_corr
        self.true_theta = true_theta
        self.true_theta_sampled = true_theta_sampled
    def inference_test_run(self):
        '''
        Load model for inference
        '''
        if self.model_emulator_bool== True:
            hmc_inf = NN_HMC_X(self.v_bins, self.best_params, self.T0s, self.gammas, self.fobs, self.like_dict, dense_mass=True,
                        max_tree_depth= 10,
                        num_warmup=1000,
                        num_samples=1000,
                        num_chains=4)
        else:
            in_name_new_params = f'new_covariances_dict_R_30000_nf_9_ncovar_{self.n_covar}_' \
                                 f'P{self.n_path}{self.bin_label}_params.p'
            new_param_dict = dill.load(open(self.in_path + in_name_new_params, 'rb'))
            new_temps = new_param_dict['new_temps']
            new_gammas = new_param_dict['new_gammas']
            new_fobs = new_param_dict['new_fobs']

            n_new_t = (len(new_temps) - 1) / (len(self.T0s) - 1) - 1
            n_new_g = (len(new_gammas) - 1) / (len(self.gammas) - 1) - 1
            n_new_f = (len(new_fobs) - 1) / (len(self.fobs) - 1) - 1
            new_models_np = np.empty([len(new_temps), len(new_gammas), len(new_fobs), len(self.v_bins)])
            new_covariances_np = np.empty([len(new_temps), len(new_gammas), len(new_fobs), len(self.v_bins), len(self.v_bins)])
            new_log_dets_np = np.empty([len(new_temps), len(new_gammas), len(new_fobs)])

            for old_t_below_idx in range(self.n_temps - 1):
                print(f'{old_t_below_idx / (self.n_temps - 1) * 100}%')
                for old_g_below_idx in range(self.n_gammas - 1):
                    for old_f_below_idx in range(self.n_f - 1):
                        fine_dict_in_name = f'new_covariances_dict_R_{int(self.R_value)}_nf_{self.n_f}_T{old_t_below_idx}_' \
                                            f'G{old_g_below_idx}_SNR0_F{old_f_below_idx}_ncovar_{self.n_covar}_' \
                                            f'P{self.n_path}{self.bin_label}.p'
                        fine_dict = dill.load(open(self.in_path + fine_dict_in_name, 'rb'))
                        new_temps_small = fine_dict['new_temps']
                        new_gammas_small = fine_dict['new_gammas']
                        new_fobs_small = fine_dict['new_fobs']
                        new_models_small = fine_dict['new_models']
                        new_covariances_small = fine_dict['new_covariances']
                        new_log_dets_small = fine_dict['new_log_dets']
                        if old_t_below_idx == self.n_temps - 2:
                            added_t_range = n_new_t + 2
                        else:
                            added_t_range = n_new_t + 1
                        if old_g_below_idx == self.n_gammas - 2:
                            added_g_range = n_new_g + 2
                        else:
                            added_g_range = n_new_g + 1
                        if old_f_below_idx == self.n_f - 2:
                            added_f_range = n_new_f + 2
                        else:
                            added_f_range = n_new_f + 1
                        for added_t_idx in range(int(added_t_range)):
                            for added_g_idx in range(int(added_g_range)):
                                for added_f_idx in range(int(added_f_range)):
                                    final_t_idx = int((old_t_below_idx * (n_new_t + 1)) + added_t_idx)
                                    final_g_idx = int((old_g_below_idx * (n_new_g + 1)) + added_g_idx)
                                    final_f_idx = int((old_f_below_idx * (n_new_f + 1)) + added_f_idx)
                                    new_models_np[final_t_idx, final_g_idx, final_f_idx, :] = new_models_small[
                                                                                              added_t_idx, added_g_idx,
                                                                                              added_f_idx, :]
                                    new_covariances_np[final_t_idx, final_g_idx, final_f_idx, :,
                                    :] = new_covariances_small[added_t_idx, added_g_idx, added_f_idx, :, :]
                                    new_log_dets_np[final_t_idx, final_g_idx, final_f_idx] = new_log_dets_small[
                                        added_t_idx, added_g_idx, added_f_idx]
            new_models = jnp.array(new_models_np)
            new_covariances = jnp.array(new_covariances_np)
            new_log_dets = jnp.array(new_log_dets_np)
            #hmc_inf = HMC_NGP(self.v_bins, new_temps_small, new_gammas_small, new_fobs_small, new_models, new_covariances, new_log_dets)
            hmc_inf = HMC_NGP(self.v_bins, new_temps, new_gammas, new_fobs, new_models, new_covariances, new_log_dets)

        '''
        File names for saving
        '''
        ### change this to the correct path ###
        if self.model_emulator_bool:
            if self.emu_test_bool:
                out_path_plot = f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{self.z_string}/emu_infer/'
            else:
                out_path_plot = f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{self.z_string}/mock_infer/'
        else:
            out_path_plot = f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{self.z_string}/ngp_infer/'
        out_path = self.out_path

        ### If the true LogP is NGP or on prior: Boundary problem ###
        if self.true_log_prob_on_prior:
            self.save_name = f"{self.out_tag}_true_theta_sampled_inference_{self.n_inference}_{self.note}_samples_{hmc_inf.num_samples}_chains_{hmc_inf.num_chains}_{self.var_tag}"
        else:
            self.save_name = f"{self.out_tag}_inference_{self.n_inference}_{self.note}_samples_{hmc_inf.num_samples}_chains_{hmc_inf.num_chains}_{self.var_tag}"

        '''
        True and Infered models
        '''
        #getting ready for inference samples
        key = random.PRNGKey(self.key_hmc)
        infer_theta = np.empty([self.n_inference, self.n_params])
        log_prob = np.empty([self.n_inference, hmc_inf.num_samples*hmc_inf.num_chains])
        true_log_prob = np.empty([self.n_inference])
        samples = np.empty([self.n_inference, hmc_inf.num_samples*hmc_inf.num_chains, self.n_params])

        #read in true mocks
        true_theta = self.true_theta
        true_theta_sampled = self.true_theta_sampled
        mocks = self.mock_corr
        covars = self.mock_covar #covariance matrix for each mock in emulator using ngp

        '''
        Run inference test for each mock
        '''
        var_label = ['fobs', 'T0s', 'gammas']
        pbar = ProgressBar()
        print(f'Start {self.n_inference} inference test for:{self.save_name}')
        for mock_idx in pbar(range(self.n_inference)):
            key, subkey = random.split(key)

            closest_temp_idx = np.argmin(np.abs(self.T0s - true_theta[mock_idx, 1]))
            closest_gamma_idx = np.argmin(np.abs(self.gammas - true_theta[mock_idx, 2]))
            closest_fobs_idx = np.argmin(np.abs(self.fobs - true_theta[mock_idx, 0]))

            if self.true_log_prob_on_prior:
                x_true = hmc_inf.theta_to_x(true_theta_sampled[mock_idx, :])
            else:
                x_true = hmc_inf.theta_to_x(true_theta[mock_idx, :])
            flux = mocks[mock_idx, :]

            #Use one coarse grid NGP covariance matrix for both NGP & Emulator models
            covars_mock = covars[mock_idx, :, :]


            x_samples, theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean, \
            hmc_num_steps, hmc_tree_depth, total_time = hmc_inf.mcmc_one(subkey, x_true, flux, covars_mock, report=False)
            f_mcmc, t_mcmc, g_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                         zip(*np.percentile(theta_samples, [16, 50, 84], axis=0)))

            infer_theta[mock_idx, :] = [f_mcmc[0], t_mcmc[0], g_mcmc[0]]
            samples[mock_idx, :, :] = theta_samples
            log_prob[mock_idx, :] = lnP
            true_log_prob[mock_idx] = -1 * hmc_inf.potential_fun(x_true, flux, covars_mock)
            #corner plot for each inference
            if mock_idx < 10:
                corner_fig = corner.corner(np.array(theta_samples), levels=(0.68, 0.95), labels=var_label,
                                           truths=np.array(true_theta[mock_idx, :]), truth_color='red', show_titles=True,
                                           quantiles=(0.16, 0.5, 0.84),title_kwargs={"fontsize": 15}, label_kwargs={'fontsize': 15},
                                           data_kwargs={'ms': 1.0, 'alpha': 0.1}, hist_kwargs=dict(density=True))
                fit_fig =  hmc_inf.fit_plot(z_string='z54',best_params=self.best_params,theta_samples=theta_samples, lnP = lnP,
                                            theta_true=true_theta[mock_idx, :],model_corr=self.model_corr[mock_idx, :],mock_corr=flux,
                                            infer_model= emu.nn_emulator(self.best_params, infer_theta[mock_idx, :]),covariance=covars_mock)
                if self.true_log_prob_on_prior:
                    corner_fig.savefig(out_path_plot + f'corner_T{closest_temp_idx}_G{closest_gamma_idx}_SNR{self.noise_idx}_F{closest_fobs_idx}_P{self.n_path}{self.bin_label}_mock_{mock_idx}_{self.var_tag}_{self.note}_true_theta_sampled.png')
                    fit_fig.savefig(out_path_plot + f'fit_T{closest_temp_idx}_G{closest_gamma_idx}_SNR{self.noise_idx}_F{closest_fobs_idx}_P{self.n_path}{self.bin_label}_mock_{mock_idx}_{self.var_tag}_{self.note}_true_theta_sampled.png')
                else:
                    corner_fig.savefig(
                        out_path_plot + f'corner_T{closest_temp_idx}_G{closest_gamma_idx}_SNR{self.noise_idx}_F{closest_fobs_idx}_P{self.n_path}{self.bin_label}_mock_{mock_idx}_{self.var_tag}_{self.note}_ngp.png')
                    fit_fig.savefig(out_path_plot + f'fit_T{closest_temp_idx}_G{closest_gamma_idx}_SNR{self.noise_idx}_F{closest_fobs_idx}_P{self.n_path}{self.bin_label}_mock_{mock_idx}_{self.var_tag}_{self.note}_ngp.png')

        self.infer_theta = infer_theta
        self.log_prob_x = log_prob
        self.true_log_prob_x = true_log_prob
        self.samples_theta = samples
        self.out_path = out_path
        self.out_path_plot = out_path_plot
        if self.model_emulator_bool:
            self.infer_model = emu.nn_emulator(self.best_params, self.infer_theta)
        else:
            self.infer_model = None #hmc_inf.get_model_nearest_fine(self.infer_theta)
        '''
        plot HMC inference test results
        '''
    def coverage_plot(self):
        print('plotting')

        alpha_vec = np.concatenate((np.linspace(0.00, 0.994, num=100), np.linspace(0.995, 1.0, num=51)))
        coverage_gauss, coverage_gauss_lo, coverage_gauss_hi = run_inference_test(alpha_vec, self.log_prob_x, self.true_log_prob_x,
                                                                                  title='Gaussian Lhood', show=False)

        inference_fig = plt.figure(figsize=(x_size, x_size * .9), constrained_layout=True,
                                   dpi=dpi_value,
                                   )
        grid = inference_fig.add_gridspec(
            nrows=1, ncols=1,
        )

        skew_ax = inference_fig.add_subplot(grid[0])

        skew_ax.plot(alpha_vec, coverage_gauss, color='black', linestyle='solid', label='inference test points',
                     zorder=10)
        skew_ax.fill_between(alpha_vec, coverage_gauss_lo, coverage_gauss_hi, facecolor='grey', alpha=0.8, zorder=3)
        x_vec = np.linspace(0.0, 1.0, 11)
        skew_ax.plot(x_vec, x_vec, linewidth=1.5, color='red', linestyle=(0, (5, 10)), zorder=20, label='inferred model')

        skew_ax.set_xlim((-0.01, 1.01))
        skew_ax.set_ylim((-0.01, 1.01))
        skew_ax.set_xlabel(r'$P_{{\rm true}}$', fontsize=16)
        skew_ax.set_ylabel(r'$P_{{\rm inf}}$', fontsize=16)

        inference_fig.suptitle(f'{self.note}')
        inference_fig.savefig(self.out_path_plot + f'{self.save_name}.png')
        print(f'plot saved as: {self.out_path_plot}{self.save_name}.png')

        #save HMC inference results
        with h5py.File(self.out_path + f'{self.save_name}.hdf5', 'a') as f:
            f.create_dataset('true_theta', data=self.true_theta)
            f.create_dataset('log_prob_x', data=self.log_prob_x)
            f.create_dataset('true_log_prob_x', data=self.true_log_prob_x)
            f.create_dataset('samples_theta', data=self.true_theta_sampled)
            f.create_dataset('infer_theta', data=self.infer_theta)
            f.create_dataset('inferred_model', data=self.infer_model)
            f.create_dataset('model_corr', data=self.model_corr)
            f.create_dataset('mock_corr', data=self.mock_corr)
            f.create_dataset('mock_covar', data=self.mock_covar)
        print(f'Inference test results saved as {self.out_path}{self.save_name}.hdf5 saved')


#IPython.embed()
'''
##emulator -- emulator model test
'''
hmc_infer = INFERENCE_TEST(redshift=5.4,model_emulator_bool=True,gaussian_bool=True,ngp_bool=True,emu_test_bool=True,n_inference=500,key_sample=222,key_hmc=42)

'''
##forward mocks -- emulator model
'''
#hmc_infer = INFERENCE_TEST(redshift=5.4,model_emulator_bool=True,gaussian_bool=False,ngp_bool=True,emu_test_bool=False,n_inference=100)

'''
##gaussian mocks -- NGP model
'''

#hmc_infer = INFERENCE_TEST(5.4,False,True,True,False)
#hmc_infer = INFERENCE_TEST(5.4,False,False,True,False)


hmc_infer.mocks_sampling()
hmc_infer.inference_test_run()
hmc_infer.coverage_plot()