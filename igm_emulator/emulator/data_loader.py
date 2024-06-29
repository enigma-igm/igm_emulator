import sys
import os
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/scripts')
from grab_models import param_transform
#sys.path.append(os.path.expanduser('~') + '/LAF_emulator')
#from laf_emulator.emulators.lhc_for_amber import corner_plot
import dill
import os
import numpy as np
from matplotlib import pyplot as plt
import corner
import tensorflow as tf
from sklearn.model_selection import train_test_split
import h5py
import IPython
import jax
import jax.numpy as jnp
import jax.random as random

fig = {'font.family': 'serif'
       }
plt.rcParams.update(fig)

class DataSamplerModule:
    def __init__(self, redshift=5.5,
                small_bin_bool=True,
                n_f=4, n_t=7, n_g=4,
                n_testing=0,
                seed=42,
                plot_bool=True):
        '''


        Parameters
        ----------
        redshift: float, redshift of the data [5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]
        small_bin_bool
        n_f: int, number of flux bins for emulator in total
        n_t: int, number of temperature bins for emulator in total
        n_g: int, number of gamma bins for emulator in total
        n_testing: int, number of testing samples apart from the regular grid testing selection (to keep train+validation fixed)
        seed_err
        seed_train
        plot_bool

        Returns
        -------
        z_string: str, redshift string
        in_path: str, path to the data
        all_data: np.array, all data
        fobs: np.array, average observed flux <F> ~ Gamma_HI -9
        T0s: np.array, log(T_0) from temperature - density relation -15
        gammas: np.array, gamma from temperature - density relation -9

        '''

        super().__init__()

        self.redshift = redshift
        self.small_bin_bool = small_bin_bool
        self.n_f = n_f
        self.n_t = n_t
        self.n_g = n_g
        self.n_testing = n_testing
        self.seed = seed
        self.plot_bool = plot_bool


        # get the appropriate string and pathlength for chosen redshift
        zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
        z_idx = np.argmin(np.abs(zs - self.redshift))
        z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
        z_string = z_strings[z_idx]
        self.z_string = z_string
        n_paths = np.array([17, 16, 16, 15, 15, 15, 14])

        if self.small_bin_bool:
            #smaller bins
            self.n_path = 20
            self.n_covar = 500000
            self.bin_label = '_set_bins_3'
            self.in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final_135/{z_string}/'
            in_name_h5py = f'correlation_temp_fluct_skewers_2000_R_30000_nf_9_dict_set_bins_3.hdf5'
            with h5py.File(self.in_path + in_name_h5py, 'r') as f:
                param_dict = dict(f['params'].attrs.items())
            self.fobs = param_dict['average_observed_flux']
            self.T0s = 10. ** param_dict['logT_0']
            self.gammas = param_dict['gamma']
        else:
            #larger bins
            self.n_path = n_paths[z_idx]
            self.n_covar = 500000
            self.bin_label = '_set_bins_4'
            self.in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'
            param_in_path = '/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/'
            self.param_dict = dill.load(open(param_in_path + f'{z_string}_params.p', 'rb'))
            self.fobs = param_dict['fobs']  # average observed flux <F> ~ Gamma_HI -9
            log_T0s = param_dict['log_T0s']  # log(T_0) from temperature - density relation -15
            self.T0s = np.power(10,log_T0s)
            self.gammas = param_dict['gammas']  # gamma from temperature - density relation -9

        print(f'Loading parameter grid for redshift {self.redshift}...')
        print(f'fobs:{self.fobs}')
        print(f'T0s: {self.T0s}')
        print(f'gammas:{self.gammas}')

        # Construct all data
        self.xv, self.yv, self.zv = np.meshgrid(self.fobs, self.T0s, self.gammas) # all in physical grids
        all_data = np.array([self.xv.flatten(),self.yv.flatten(), self.zv.flatten()])
        self.all_data = all_data.T
        # all_data.shape = (1215, 3)



    def regular_grid(self):

        # Construct regular grid for training
        x = np.linspace(0,1,self.n_f)
        y = np.linspace(0,1,self.n_t)
        z = np.linspace(0,1,self.n_g)
        n_samples = x.shape[0]*y.shape[0]*z.shape[0] #n_sample = 768
        final_samples = np.empty([n_samples, 3])
        xg, yg, zg = np.meshgrid(x, y, z)


        # convert the output of grid (between 0 and 1 for each parameter) to our model grid
        if self.redshift >= 5.9:
            print('discard smallest flux bin')
            xg_trans = param_transform(xg, self.fobs[1], self.fobs[-1]) #9 --> 8 for z = 5.9, 6.0
        else:
            xg_trans = param_transform(xg, self.fobs[0], self.fobs[-1]) #9
        yg_trans = param_transform(yg, self.T0s[0], self.T0s[-1]) #15
        zg_trans = param_transform(zg, self.gammas[0], self.gammas[-1]) #9
        sample_params = np.array([xg_trans.flatten(),yg_trans.flatten(),zg_trans.flatten()])
        sample_params = sample_params.T


        for sample_idx in np.arange(n_samples):

            sample = sample_params[sample_idx]

            # determine the closest model to each lhs sample
            fobs_idx = np.argmin(np.abs(self.fobs - sample[0]))
            T0_idx = np.argmin(np.abs(self.T0s - sample[1]))
            gamma_idx = np.argmin(np.abs(self.gammas - sample[2]))

            # save the closest model parameters for each lhs sample
            final_samples[sample_idx, 0] = self.fobs[fobs_idx]
            final_samples[sample_idx, 1] = self.T0s[T0_idx]
            final_samples[sample_idx, 2] = self.gammas[gamma_idx]

            # get the corresponding model autocorrelation for each parameter location
            # **smaller bins**
            if self.small_bin_bool:
                like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{gamma_idx}_SNR0_F{fobs_idx}_ncovar_500000_P{self.n_path}_set_bins_3.p'

            # **larger bins**
            else:
                like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{gamma_idx}_SNR0_F{fobs_idx}_ncovar_500000_P{self.n_path}_set_bins_4.p'

            like_dict = dill.load(open(self.in_path + like_name, 'rb'))
            model_autocorrelation = like_dict['mean_data']
            if sample_idx == 0:
                models = np.empty([n_samples, len(model_autocorrelation)])
            models[sample_idx] = model_autocorrelation

        # Filter out repeated data
        final_params = [] #training dataset
        count = 0
        for idx, data in enumerate(final_samples):
            for i in final_params:
                if np.array_equal(data,i):
                    count += 1
            if count == 0:
                final_params.append(data)
            count = 0
        final_samples = np.asarray(final_params) #(768, 3)


        # Test data
        test_param = []
        test_corr = []
        for idx, data in enumerate(self.all_data):
            count = 0
            for i in final_samples:
                if np.array_equal(data,i):
                    count += 1
                    break
                else:
                    pass
            if count == 0:
                test_param.append(data)

                fobs_idx = np.argmin(np.abs(self.fobs - data[0]))
                T0_idx = np.argmin(np.abs(self.T0s - data[1]))
                gamma_idx = np.argmin(np.abs(self.gammas - data[2]))

                # get the corresponding model autocorrelation for each parameter location
                #smaller bins
                if self.small_bin_bool:
                    like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{gamma_idx}_SNR0_F{fobs_idx}_ncovar_500000_P{self.n_path}_set_bins_3.p'
                #larger bins
                else:
                    like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{gamma_idx}_SNR0_F{fobs_idx}_ncovar_500000_P{self.n_path}_set_bins_4.p'
                like_dict = dill.load(open(self.in_path + like_name, 'rb'))
                model_autocorrelation = like_dict['mean_data']
                test_corr.append(model_autocorrelation)
        test_param = np.asarray(test_param)
        test_corr = np.asarray(test_corr)


        n_validation = test_corr.shape[0] - self.n_testing
        test_selection = tf.random.shuffle([True]*self.n_testing+[False]*n_validation)#, lambda:random.gauss(0.5,0.1))

        vali_param, vali_corr = test_param[~test_selection], test_corr[~test_selection] #validation dataset = (358, )
        testing_param, testing_corr = test_param[test_selection], test_corr[test_selection] #testing dataset = (89, )

        return final_samples,testing_param,testing_corr

    def corner_plot(self, data, labels, hist_bins=30, scatter_alpha=0.5, hist_alpha=0.5, fig_name=None, point_color='k'):

        num_dimensions = data.shape[1]

        fig, axes = plt.subplots(nrows=num_dimensions, ncols=num_dimensions, figsize=(10, 10))

        for i in range(num_dimensions):
            for j in range(num_dimensions):
                if i == j:
                    axes[i, j].hist(data[:, i], bins=hist_bins, alpha=hist_alpha, color=point_color)
                    axes[i, j].set_ylabel('')
                    axes[i, j].yaxis.set_ticklabels([])  # Remove y-axis labels for histograms
                    if i == num_dimensions - 1:  # Last row
                        axes[i, j].set_xlabel(labels[i])
                        axes[i, j].xaxis.set_tick_params(which='both', labelbottom=True)
                    else:
                        axes[i, j].set_xlabel('')
                        axes[i, j].xaxis.set_tick_params(which='both', labelbottom=False)
                elif i > j:  # Only plot scatter plots below the diagonal
                    axes[i, j].scatter(data[:, j], data[:, i], alpha=scatter_alpha, color=point_color)
                    axes[i, j].set_xlabel(labels[j])
                    axes[i, j].set_ylabel(labels[i])
                    if i == num_dimensions - 1:  # Last row
                        axes[i, j].set_xlabel(labels[j])
                        axes[i, j].xaxis.set_tick_params(which='both', labelbottom=True)
                    else:
                        axes[i, j].set_xlabel('')
                        axes[i, j].xaxis.set_tick_params(which='both', labelbottom=False)
                    if j == 0:  # First column
                        axes[i, j].set_ylabel(labels[i])
                        axes[i, j].yaxis.set_tick_params(which='both', labelleft=True)
                    else:
                        axes[i, j].set_ylabel('')
                        axes[i, j].yaxis.set_tick_params(which='both', labelleft=False)
                else:
                    axes[i, j].axis('off')  # Turn off plots above the diagonal

        plt.tight_layout()

        if fig_name is not None:
            plt.savefig(fig_name, format='png', dpi=512)
        else:
            plt.show()

        plt.close(fig)
    def random_split(self, sparce_samples, test_size=0.1, train_size=0.5):
        '''

        Parameters
        ----------
        seed:Integer values must be in the range [0, 2**32 - 1].

        Returns
        -------

        '''
        X_train_vali, X_test = train_test_split(sparce_samples, train_size=1-test_size, random_state=self.seed)
        X_train, X_vali = train_test_split(X_train_vali, train_size=train_size/(1-test_size), random_state=self.seed)
        self.X_train = X_train
        self.X_test = X_test
        self.X_vali = X_vali
        train_corr = []
        test_corr = []
        vali_corr = []

        for idx, data in enumerate(X_train):
            fobs_idx = np.argmin(np.abs(self.fobs - data[0]))
            T0_idx = np.argmin(np.abs(self.T0s - data[1]))
            gamma_idx = np.argmin(np.abs(self.gammas - data[2]))

            # get the corresponding model autocorrelation for each parameter location
            # **smaller bins**
            if self.small_bin_bool:
                like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{gamma_idx}_SNR0_F{fobs_idx}_ncovar_500000_P{self.n_path}_set_bins_3.p'
            else:
                like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{gamma_idx}_SNR0_F{fobs_idx}_ncovar_500000_P{self.n_path}_set_bins_4.p'
            like_dict = dill.load(open(self.in_path + like_name, 'rb'))
            train_corr.append(like_dict['mean_data'])
        self.train_corr = np.asarray(train_corr)

        for idx, data in enumerate(X_test):
            fobs_idx = np.argmin(np.abs(self.fobs - data[0]))
            T0_idx = np.argmin(np.abs(self.T0s - data[1]))
            gamma_idx = np.argmin(np.abs(self.gammas - data[2]))


            # get the corresponding model autocorrelation for each parameter location
            # **smaller bins**
            if self.small_bin_bool:
                like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{gamma_idx}_SNR0_F{fobs_idx}_ncovar_500000_P{self.n_path}_set_bins_3.p'
            else:
                like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{gamma_idx}_SNR0_F{fobs_idx}_ncovar_500000_P{self.n_path}_set_bins_4.p'
            like_dict = dill.load(open(self.in_path + like_name, 'rb'))
            test_corr.append(like_dict['mean_data'])
        self.test_corr = np.asarray(test_corr)

        for idx, data in enumerate(X_vali):
            fobs_idx = np.argmin(np.abs(self.fobs - data[0]))
            T0_idx = np.argmin(np.abs(self.T0s - data[1]))
            gamma_idx = np.argmin(np.abs(self.gammas - data[2]))

            # get the corresponding model autocorrelation for each parameter location
            # **smaller bins**
            if self.small_bin_bool:
                like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{gamma_idx}_SNR0_F{fobs_idx}_ncovar_500000_P{self.n_path}_set_bins_3.p'
            else:
                like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{gamma_idx}_SNR0_F{fobs_idx}_ncovar_500000_P{self.n_path}_set_bins_4.p'
            like_dict = dill.load(open(self.in_path + like_name, 'rb'))
            vali_corr.append(like_dict['mean_data'])
        self.vali_corr = np.asarray(vali_corr)


    def data_sampler(self):
        self.sparce_samples,self.extra_test_err_param,self.extra_test_err_corr = self.regular_grid()
        self.random_split(self.sparce_samples, test_size=0.1, train_size=0.5)

        dir = '/home/zhenyujin/igm_emulator/igm_emulator/emulator/GRID'
        dir_plot = f'/mnt/quasar2/zhenyujin/igm_emulator/emulator/plots/{self.z_string}'

        if self.plot_bool:
            H = self.X_train
            ax = plt.axes(projection='3d')
            ax.scatter(self.xv, self.yv, self.zv, c='b', alpha=0.1, linewidth=0.5, label=f'all data: {self.all_data.shape[0]}')
            ax.scatter(H[:, 0], H[:, 1], H[:, 2], c='r', linewidth=0.2, label=f'training data: {H.shape[0]}')
            A = self.X_vali
            ax.scatter(A[:, 0], A[:, 1], A[:, 2], c='g', linewidth=0.2, label=f'validation data: {A.shape[0]}')
            T = self.X_test
            T_err = self.extra_test_err_param
            ax.scatter(T[:, 0], T[:, 1], T[:, 2], c='k', linewidth=0.2, label=f'testing data: {T.shape[0]+T_err.shape[0]}')
            ax.scatter(T_err[:, 0], T_err[:, 1], T_err[:, 2], c='k', linewidth=0.2, alpha = 0.5)
            ax.set_xlabel(r'$<F>$')
            ax.set_ylabel(r'$T_0$')
            ax.set_zlabel(r'$\gamma$')
            ax.legend()
            ax.grid(True)
            plt.savefig(os.path.join(dir_plot,f"{self.z_string}_params_sampling_random_split_train_{self.X_train.shape[0]}_test_{T.shape[0]+T_err.shape[0]}_seed_{self.seed}.png"))
            plt.close()
            self.corner_plot(H, labels= [r'$\langle F \rangle$', r'$T_0$', r'$\gamma$'], fig_name=f'{dir_plot}/corner_plot_train_{self.X_train.shape[0]}_seed_{self.seed}_params_loaded.png')

        if self.small_bin_bool:
            num = f'bin59_seed_{self.seed}' #if seed = None, it's regular grid
        else:
            num = f'bin276_seed_{self.seed}'

        ### Join the extra test data to the test data -- only control test dataset size
        self.X_test = np.concatenate((self.X_test, self.extra_test_err_param), axis=0)
        self.test_corr = np.concatenate((self.test_corr, self.extra_test_err_corr), axis=0)

        train_num = f'_train_{self.X_train.shape[0]}'
        dill.dump(self.X_train,open(os.path.join(dir, f'{self.z_string}_param{train_num}_{num}.p'),'wb'))
        dill.dump(self.train_corr,open(os.path.join(dir, f'{self.z_string}_model{train_num}_{num}.p'),'wb'))

        test_num=f'_test_{self.X_test.shape[0]}'
        dill.dump(self.X_test,open(os.path.join(dir, f'{self.z_string}_param{test_num}_{num}.p'),'wb'))
        dill.dump(self.test_corr,open(os.path.join(dir, f'{self.z_string}_model{test_num}_{num}.p'),'wb'))

        vali_num=f'_vali_{self.X_vali.shape[0]}'
        dill.dump(self.X_vali,open(os.path.join(dir, f'{self.z_string}_param{vali_num}_{num}.p'),'wb'))
        dill.dump(self.vali_corr,open(os.path.join(dir, f'{self.z_string}_model{vali_num}_{num}.p'),'wb'))

        print(f'Datasets saved at {dir}')
        print(f'Train: {self.z_string}_param{train_num}_{num}.p; {self.z_string}_model{train_num}_{num}.p')
        print(f'Test: {self.z_string}_param{test_num}_{num}.p; {self.z_string}_model{test_num}_{num}.p')
        print(f'Validation: {self.z_string}_param{vali_num}_{num}.p; {self.z_string}_model{vali_num}_{num}.p')

        # get the fixed covariance dictionary for likelihood
        T0_idx = 7  # 0-14
        g_idx = 4  # 0-8
        f_idx = 4  # 0-8
        like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{g_idx}_SNR0_F{f_idx}_ncovar_{self.n_covar}_P{self.n_path}{self.bin_label}.p'
        self.like_dict = dill.load(open(self.in_path + like_name, 'rb'))
        print(f'likelihood on Fobs:{self.fobs[f_idx]}, T_0:{self.T0s[T0_idx]}, gamma:{self.gammas[g_idx]}')

        self.out_tag = f'{self.z_string}{train_num}_{num}'
        self.train_num = train_num
        self.test_num = test_num
        self.vali_num = vali_num

        return self.X_train, self.train_corr, self.X_test, self.test_corr, self.X_vali, self.vali_corr, self.like_dict