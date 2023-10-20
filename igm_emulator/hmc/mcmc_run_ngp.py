import numpy as np
from scipy.stats import multivariate_normal
class MCMC_NGP:
    def __int__(self, like_dict, new_param_dict, ):
    def get_covariance_log_determinant_nearest_fine(
            theta,
            fine_mean_free_paths, fine_average_fluxes, fine_covariances, fine_log_determinants,
    ):

        mfp, ave_f = theta

        mfp_idx_closest = np.argmin(np.abs(fine_mean_free_paths - mfp))
        f_idx_closest = np.argmin(np.abs(fine_average_fluxes - ave_f))

        covariance = fine_covariances[mfp_idx_closest, f_idx_closest, :, :]
        log_determinant = fine_log_determinants[mfp_idx_closest, f_idx_closest]

        return covariance, log_determinant

    def get_covariance_nearest_fine(
            theta,
            fine_fobs, fine_T0s, fine_gammas, fine_covariances,
    ):

        f, t, g = theta

        f_idx_closest = np.argmin(np.abs(fine_fobs - f))
        t_idx_closest = np.argmin(np.abs(fine_T0s - t))
        g_idx_closest = np.argmin(np.abs(fine_gammas - g))

        covariance = fine_covariances[f_idx_closest, t_idx_closest, g_idx_closest, :, :]
        # need to read in the fine_covariances
        return covariance

    def get_model_nearest_fine(
            theta,
            fine_mean_free_paths, fine_average_fluxes, fine_models
    ):

        mfp, ave_f = theta

        mfp_idx_closest = np.argmin(np.abs(fine_mean_free_paths - mfp))
        f_idx_closest = np.argmin(np.abs(fine_average_fluxes - ave_f))

        model = fine_models[mfp_idx_closest, f_idx_closest, :]

        return model

    def log_likelihood_old(
            theta, data_auto_correlation,
            fine_mean_free_paths, fine_average_fluxes, fine_models, fine_covariances, fine_log_determinants,
            alpha_const=None,
            fixed_covariance_bool=False,
            fixed_covariance=None, fixed_log_det=None
    ):

        model = get_model_nearest_fine(theta, fine_mean_free_paths, fine_average_fluxes, fine_models)

        if fixed_covariance_bool:
            covariance = fixed_covariance
            log_determinant = fixed_log_det
        else:
            covariance, log_determinant = get_covariance_log_determinant_nearest_fine(
                theta,
                fine_mean_free_paths, fine_average_fluxes, fine_covariances, fine_log_determinants
            )

        if alpha_const:
            covariance = covariance * alpha_const ** -1
            log_determinant = log_determinant + np.log(alpha_const ** float(len(data_auto_correlation)))

        difference = data_auto_correlation - model
        n_bins = len(data_auto_correlation)
        log_like = -(np.dot(difference, np.linalg.solve(covariance, difference)) + log_determinant + n_bins * np.log(
            2.0 * np.pi)) / 2.0

        return log_like

    def log_likelihood(
            theta, data_auto_correlation,
            fine_mean_free_paths, fine_average_fluxes, fine_models, fine_covariances,
            alpha_const=None,
            fixed_covariance_bool=False,
            fixed_covariance=None,
    ):

        model = get_model_nearest_fine(theta, fine_mean_free_paths, fine_average_fluxes, fine_models)

        if fixed_covariance_bool:
            covariance = fixed_covariance
        else:
            covariance = get_covariance_nearest_fine(
                theta,
                fine_mean_free_paths, fine_average_fluxes, fine_covariances
            )

        if alpha_const:
            covariance = covariance * alpha_const ** -1

        log_like = multivariate_normal.logpdf(data_auto_correlation, mean=model, cov=covariance)

        return log_like

    def log_prior(theta, mean_free_paths, average_fluxes):

        mfp, ave_f = theta

        if mean_free_paths[0] <= mfp <= mean_free_paths[-1] and average_fluxes[0] <= ave_f <= average_fluxes[-1]:
            # if mean_free_paths[0] <= mfp <= mean_free_paths[-1] and average_fluxes[0] <= ave_f <= 0.52:
            return 0.0
        return -np.inf

    def log_prior_gaussian_f(theta, mean_free_paths, mean_flux, error_flux):

        mfp, ave_f = theta

        if mean_free_paths[0] <= mfp <= mean_free_paths[-1]:
            return multivariate_normal.logpdf(ave_f, mean=mean_flux, cov=error_flux ** 2)
        return -np.inf

    def log_probability_old(
            theta, data_auto_correlation,
            fine_mean_free_paths, fine_average_fluxes, fine_models, fine_covariances, fine_log_determinants,
            alpha_const=None,
            fixed_covariance_bool=False,
            fixed_covariance=None, fixed_log_det=None
    ):

        lp = log_prior(theta, fine_mean_free_paths, fine_average_fluxes)
        if not np.isfinite(lp):
            return -np.inf

        return lp + log_likelihood_old(
            theta, data_auto_correlation,
            fine_mean_free_paths, fine_average_fluxes, fine_models, fine_covariances, fine_log_determinants,
            alpha_const=alpha_const,
            fixed_covariance_bool=fixed_covariance_bool,
            fixed_covariance=fixed_covariance, fixed_log_det=fixed_log_det
        )

    def log_probability(
            theta, data_auto_correlation,
            fine_mean_free_paths, fine_average_fluxes, fine_models, fine_covariances,
            alpha_const=None,
            fixed_covariance_bool=False,
            fixed_covariance=None,
            gaussian_f_prior=False,
            mean_flux=None, error_flux=None,
    ):

        if gaussian_f_prior:
            lp = log_prior_gaussian_f(theta, fine_mean_free_paths, mean_flux, error_flux)
        else:
            lp = log_prior(theta, fine_mean_free_paths, fine_average_fluxes)

        if not np.isfinite(lp):
            return -np.inf

        return lp + log_likelihood(
            theta, data_auto_correlation,
            fine_mean_free_paths, fine_average_fluxes, fine_models, fine_covariances,
            alpha_const=alpha_const,
            fixed_covariance_bool=fixed_covariance_bool,
            fixed_covariance=fixed_covariance,
        )