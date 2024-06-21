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
from igm_emulator.emulator.emulator_apply import trainer, small_bin_bool, test_num, z_string
from hmc_ngp_inference import HMC_NGP
sys.path.append('/home/zhenyujin/qso_fitting/')
import h5py
from qso_fitting.analysis.inference_test import run_inference_test, compute_importance_weights, C_ge
import corner
from progressbar import ProgressBar

class INFERENCE_TEST():
    '''
    A class to run inference test in HMC for NGP and Emulator models
    '''
    def __init__(self,
                 gaussian_bool,  #True: Gaussian sampling around mean; False: Forward mocks sampling
                 ngp_bool, #True: NGP model; False: emulator model
                 emu_test_bool=False, #True: perfect inference test with emulator mocks; False: inference test with mocks
                 nn_err_prop_bool = True,
                 n_inference=100, n_params=3,
                 out_path = '/mnt/quasar2/zhenyujin/igm_emulator/hmc/hmc_results/',
                 key_sample=36,
                 key_hmc=642
    ):

        '''
        Args:
            gaussian_bool: bool, True: Gaussian sampling around mean; False: Forward mocks sampling
            ngp_bool: bool, True: NGP model; False: Emulator model
            emu_test_bool: bool, True: perfect inference test with emulator mocks; False: inference test with real mocks
            n_inference: int, number of mocks for inference test
            n_params: int, number of parameters for inference test
            out_path: str, path to save inference test results
            key_sample: int, random key for sampling parameters
            key_hmc: int, random key for HMC inference test
        '''

        # get the appropriate string and pathlength for chosen redshift (set in hparam_tuning.py)
        self.z_string = z_string
        self.gaussian_bool = gaussian_bool
        self.ngp_bool = ngp_bool
        self.n_inference = n_inference
        self.n_params = n_params
        self.emu_test_bool = emu_test_bool
        self.nn_err_prop_bool = nn_err_prop_bool
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
        in_path_best_params = '/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/hparam_results/'
        in_path_hdf5 = '/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/'
        self.var_tag = trainer.var_tag
        self.out_tag = trainer.out_tag
        self.best_params = dill.load(open(in_path_best_params + f'{self.out_tag}_{self.var_tag}_best_param.p', 'rb')) #changed to optuna tuned best param
        self.covar_nn = dill.load(open(in_path_hdf5 + f'{self.out_tag}{test_num}_{self.var_tag}_covar_nn.p', 'rb'))
        self.err_nn = dill.load(open(in_path_hdf5 + f'{self.out_tag}{test_num}_{self.var_tag}_err_nn.p', 'rb'))
        if self.ngp_bool:
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
        '''
        Sample parameters from priors and get mock correlation functions
        Parameters
        ----------
        self

        Returns
        -------

        '''
        if self.gaussian_bool == False:
            if self.ngp_bool == True:
                self.note = 'forward_mocks_ngp'
            else:
                self.note = 'forward_mocks_emulator'
        else:
            if self.ngp_bool == True and self.emu_test_bool == False:
                self.note = 'gaussian_mocks_ngp'
            elif self.ngp_bool == False and self.emu_test_bool == True:
                self.note = 'gaussian_mocks_emulator_TEST'
            elif self.ngp_bool == False and self.emu_test_bool == False:
                self.note = 'gaussian_mocks_emulator'
            else:
                raise ValueError('Invalid combination of ngp_bool and emu_test_bool; if emu_test_bool is True, ngp_bool must be False')
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
        true_theta_ngp = np.empty([self.n_inference, self.n_params])
        pbar = ProgressBar()

        for mock_idx in pbar(range(self.n_inference)):

            closest_temp_idx = np.argmin(np.abs(self.T0s - true_theta_sampled[mock_idx, 1]))
            closest_gamma_idx = np.argmin(np.abs(self.gammas - true_theta_sampled[mock_idx, 2]))
            closest_fobs_idx = np.argmin(np.abs(self.fobs - true_theta_sampled[mock_idx, 0]))

            true_theta_ngp = [self.fobs[closest_fobs_idx], self.T0s[closest_temp_idx], self.gammas[closest_gamma_idx]]
            model_name = f'likelihood_dicts_R_30000_nf_9_T{closest_temp_idx}_G{closest_gamma_idx}_SNR0_F{closest_fobs_idx}_ncovar_{self.n_covar}_P{self.n_path}{self.bin_label}.p'
            model_dict = dill.load(open(self.in_path + model_name, 'rb'))
            model_corr[mock_idx, :] = model_dict['mean_data']
            cov = model_dict['covariance']

            if self.gaussian_bool:
                if self.emu_test_bool:
                    mean = emu.nn_emulator(self.best_params, true_theta_sampled[mock_idx, :])
                    #cov = self.like_dict['covariance']
                else:
                    mean = model_corr[mock_idx, :]

                #split rng!
                rng, init_rng = random.split(rng)

                mock_corr[mock_idx, :] = random.multivariate_normal(init_rng, mean, cov)
                mock_covar[mock_idx, :, :] = cov

            else:
                mock_name = f'mocks_R_{int(self.R_value)}_nf_{self.n_f}_T{closest_temp_idx}_G{closest_gamma_idx}_SNR{self.noise_idx}_F{closest_fobs_idx}_P{self.n_path}{self.bin_label}.p'
                mocks = dill.load(open(self.in_path + mock_name, 'rb'))
                #split rng!
                rng, init_rng = random.split(rng)

                mock_corr[mock_idx, :] = random.choice(key=init_rng, a=mocks, shape=(1,), replace=False)
                mock_covar[mock_idx, :, :] = cov

        # save get n_inference sampled parameters and mock correlation functions
        self.mock_corr = mock_corr
        self.mock_covar = mock_covar
        self.model_corr = model_corr
        self.true_theta = true_theta_sampled
        self.true_theta_ngp = true_theta_ngp
    def inference_test_run(self):
        '''
        Run HMC inference test
        Parameters
        ----------
        self

        Returns
        -------

        '''

        '''
        Load HMC inference class
        '''

        ### NGP model inference class load in###
        if self.ngp_bool == True:
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

        ### Emulator model inference class load in###
        else:
            hmc_inf = NN_HMC_X(self.v_bins, self.best_params, self.T0s, self.gammas, self.fobs,  #switch to new_temps, new_gammas, new_fobs didn't change anything
                                dense_mass=True,
                                max_tree_depth= 10,
                                num_warmup=1000,
                                num_samples=2000,
                                num_chains=4,
                                covar_nn=self.covar_nn,
                                err_nn=self.err_nn,
                                nn_err_prop = self.nn_err_prop_bool)


        '''
        File names for saving
        '''
        ### change this to the correct path ###
        if self.ngp_bool:
            out_path_plot = f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{self.z_string}/ngp_infer/'
        else:
            if self.emu_test_bool:
                out_path_plot = f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{self.z_string}/emu_infer/'
            else:
                out_path_plot = f'/mnt/quasar2/zhenyujin/igm_emulator/hmc/plots/{self.z_string}/mock_infer/'

        out_path = self.out_path
        if os.path.exists(out_path_plot) is False:
            os.makedirs(out_path_plot)
        if hmc_inf.nn_err_prop:
            self.save_name = f"{self.out_tag}_inference_{self.n_inference}_{self.note}_seed_{self.key_sample}_samples_{hmc_inf.num_samples}_chains_{hmc_inf.num_chains}_nn_err_prop_{hmc_inf.nn_err_prop}{test_num}"
        else:
            self.save_name =f"{self.out_tag}_inference_{self.n_inference}_{self.note}_seed_{self.key_sample}_samples_{hmc_inf.num_samples}_chains_{hmc_inf.num_chains}_nn_err_prop_{hmc_inf.nn_err_prop}"

        '''
        True and Infered models
        '''
        #getting ready for inference samples
        key = random.PRNGKey(self.key_hmc)
        infer_theta = np.empty([self.n_inference, self.n_params, 3])
        log_prob = np.empty([self.n_inference, hmc_inf.num_samples*hmc_inf.num_chains])
        true_log_prob = np.empty([self.n_inference])
        samples = np.empty([self.n_inference, hmc_inf.num_samples*hmc_inf.num_chains, self.n_params])

        #read in true mocks
        true_theta = self.true_theta
        true_theta_ngp = self.true_theta_ngp
        mocks = self.mock_corr
        covars = self.mock_covar #covariance matrix for each mock in emulator using ngp
        infer_model = np.empty([self.n_inference, len(self.v_bins)])

        '''
        Run inference test for each mock
        '''
        var_label = ["<F>", "$T_0$", "$\gamma$"]
        pbar = ProgressBar()
        print(f'Start {self.n_inference} inference test for:{self.save_name}')
        for mock_idx in pbar(range(self.n_inference)):
            key, subkey = random.split(key)

            closest_temp_idx = np.argmin(np.abs(self.T0s - true_theta[mock_idx, 1]))
            closest_gamma_idx = np.argmin(np.abs(self.gammas - true_theta[mock_idx, 2]))
            closest_fobs_idx = np.argmin(np.abs(self.fobs - true_theta[mock_idx, 0]))


            x_true = hmc_inf.theta_to_x(true_theta[mock_idx, :])


            flux = mocks[mock_idx, :]

            #Use one coarse grid NGP covariance matrix for both NGP & Emulator models
            covars_mock = covars[mock_idx, :, :]

            x_opt, theta_opt, losses = hmc_inf.fit_one(flux, covars_mock)


            x_samples, theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean, \
            hmc_num_steps, hmc_tree_depth, total_time = hmc_inf.mcmc_one(subkey, x_opt, flux, covars_mock, report=False) #use x_opt instead of x_true
            f_mcmc, t_mcmc, g_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                                         zip(*np.percentile(theta_samples, [16, 50, 84], axis=0)))

            infer_theta[mock_idx, :, :] = [f_mcmc, t_mcmc, g_mcmc]
            infer_model[mock_idx, :] = hmc_inf.get_model_nearest_fine([f_mcmc[0], t_mcmc[0], g_mcmc[0]])

            samples[mock_idx, :, :] = theta_samples
            log_prob[mock_idx, :] = lnP
            true_log_prob[mock_idx] = -1 * hmc_inf.potential_fun(x_true, flux, covars_mock)

            #corner plot for each inference
            if mock_idx < 10:
                corner_fig = hmc_inf.corner_plot(self.z_string,theta_samples,x_samples,true_theta[mock_idx, :],save_str=None, save_bool=False)
                corner_fig.text(0.5, 0.77,
                                      f"opt theta: {np.array2string(theta_opt, precision=2, floatmode='fixed')}",
                                      {'fontsize': 5, 'color': 'green'})
                corner.overplot_lines(corner_fig, theta_opt, color="g")
                fit_fig =  hmc_inf.fit_plot(z_string=self.z_string,theta_samples=theta_samples, lnP = lnP,
                                            theta_true=true_theta[mock_idx, :],model_corr=self.model_corr[mock_idx, :],mock_corr=flux,
                                            covariance=covars_mock)

                corner_fig.savefig(out_path_plot + f'corner_T{closest_temp_idx}_G{closest_gamma_idx}_F{closest_fobs_idx}_mock_{mock_idx}_{self.save_name}.png')
                fit_fig.savefig(out_path_plot + f'fit_T{closest_temp_idx}_G{closest_gamma_idx}_F{closest_fobs_idx}_mock_{mock_idx}_{self.save_name}.png')
                plt.close(corner_fig)
                plt.close(fit_fig)

        self.infer_theta = infer_theta
        self.log_prob_x = log_prob
        self.true_log_prob_x = true_log_prob
        self.sampled_theta = samples
        self.out_path = out_path
        self.out_path_plot = out_path_plot
        self.infer_model = infer_model
        '''
        plot HMC inference test results
        '''
    def coverage_plot(self):
        # save HMC inference results
        if os.path.exists(self.out_path + f'{self.save_name}.hdf5'):
            os.remove(self.out_path + f'{self.save_name}.hdf5') # remove old file to rewrite
        with h5py.File(self.out_path + f'{self.save_name}.hdf5', 'a') as f:
            f.get('true_theta') or f.create_dataset('true_theta', data=self.true_theta)
            f.get('true_theta_ngp') or f.create_dataset('true_theta_ngp', data=self.true_theta_ngp)
            f.get('log_prob_x') or f.create_dataset('log_prob_x', data=self.log_prob_x)
            f.get('true_log_prob_x') or f.create_dataset('true_log_prob_x', data=self.true_log_prob_x)
            f.get('sampled_theta') or f.create_dataset('sampled_theta', data=self.sampled_theta)
            f.get('infer_theta_delta') or f.create_dataset('infer_theta_delta', data=self.infer_theta)
            f.get('inferred_model') or f.create_dataset('inferred_model', data=self.infer_model)
            f.get('model_corr') or f.create_dataset('model_corr', data=self.model_corr)
            f.get('mock_corr') or f.create_dataset('mock_corr', data=self.mock_corr)
            f.get('mock_covar') or f.create_dataset('mock_covar', data=self.mock_covar)

        print(f'Inference test results saved as {self.out_path}{self.save_name}.hdf5 saved')

        print('plotting')

        alpha_vec = np.concatenate((np.linspace(0.00, 0.994, num=100), np.linspace(0.995, 1.0, num=51)))
        coverage_gauss, coverage_gauss_lo, coverage_gauss_hi = run_inference_test(alpha_vec, self.log_prob_x, self.true_log_prob_x,
                                                                                   show=True, title=f'{self.note}',outfile=self.out_path_plot + f'{self.save_name}.png')
        print(f'plot saved as: {self.out_path_plot}{self.save_name}.png')


#IPython.embed()
'''
##emulator -- emulator model test
'''
#hmc_infer = INFERENCE_TEST(gaussian_bool=True,ngp_bool=False,emu_test_bool=True,nn_err_prop_bool=True, n_inference=100) #,key_sample=42,key_hmc=66)

'''
##forward mocks -- emulator model
'''
hmc_infer = INFERENCE_TEST(gaussian_bool=False,ngp_bool=False,emu_test_bool=False,nn_err_prop_bool=True,n_inference=100)#,key_sample=42,key_hmc=66)
#hmc_infer = INFERENCE_TEST(gaussian_bool=True,ngp_bool=False,emu_test_bool=False,nn_err_prop_bool=True, n_inference=100)#,key_sample=42,key_hmc=66)

'''
##gaussian mocks -- NGP model
'''

#hmc_infer = INFERENCE_TEST(True,True,False) #,key_sample=42,key_hmc=66)
#hmc_infer = INFERENCE_TEST(False,True,False,key_sample=42,key_hmc=66)


hmc_infer.mocks_sampling()
hmc_infer.inference_test_run()
hmc_infer.coverage_plot()
