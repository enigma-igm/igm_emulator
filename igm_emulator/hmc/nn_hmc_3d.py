import json
from jax.config import config
config.update("jax_enable_x64", True)
import h5py
from igm_emulator.scripts.pytree_h5py import save, load
import matplotlib
import matplotlib.pyplot as plt
from numpyro.infer import MCMC, NUTS
import arviz as az
import time
from IPython import embed
from dw_inference.inference.utils import walker_plot, corner_plot

redshift = 5.4
f = h5py.File(f'/home/zhenyujin/igm_emulator/igm_emulator/emulator/best_params/z{redshift}_nn_savefile.hdf5', 'r')
print(f['performance']['residuals'])
print(f['best_params']['custom_linear/~/linear_0']['w'])
print(load(f'/home/zhenyujin/igm_emulator/igm_emulator/emulator/best_params/z{redshift}_nn_savefile.hdf5'))


@partial(jit, static_argnums=(0,))
def compute_loss(x, flux, ivar, gpm_indx):
    lnPrior = eval_prior(x)
    x_astro, x_s_DR = self.param_split(x)
    s_DR = self.eval_s_DR(x_s_DR) #shape (nspec,) best fit data reduced intrinsic quasar spectrum
    lnlike = self.lnlike_split(x_astro, flux, ivar, s_DR, gpm_indx) if self.cont_momentfile is None \
        else self.lnlike_full(x_astro, flux, ivar, s_DR, gpm_indx)
    lnP = lnlike + lnPrior

    return -lnP, s_DR

@partial(jit, static_argnums=(0,))
def potential_fun(x, fslux, ivar):
    return compute_los(x, flux, ivar)[0]

def numpyro_potential_fun(flux, ivar):
    return partial(potential_fun, flux=flux, ivar=ivar)

def mcmc_one(key, x_opt, flux, ivar):
    # Instantiate the NUTS kernel and the mcmc object
    # Original line
    nuts_kernel = NUTS(potential_fn=numpyro_potential_fun(flux, ivar),
                       adapt_step_size=True, dense_mass=mcmc_dense_mass, max_tree_depth=mcmc_max_tree_depth)
    mcmc = MCMC(nuts_kernel, num_warmup=mcmc_warmup, num_samples=mcmc_nsteps, num_chains=mcmc_num_chains,
                chain_method='vectorized', jit_model_args=True)  # chain_method='sequential'
    # Run the MCMC
    start_time = time.time()
    mcmc.run(key, init_params=x_opt, extra_fields=('potential_energy', 'num_steps'))
    total_time = time.time() - start_time

    # Compute the neff and summarize cost
    az_summary = az.summary(az.from_numpyro(mcmc))
    neff = az_summary["ess_bulk"].to_numpy()
    neff_mean = np.mean(neff)
    r_hat = az_summary["r_hat"].to_numpy()
    r_hat_mean = np.mean(r_hat)
    sec_per_neff = (total_time / neff_mean)
    # Grab the samples and lnP
    x_samples = mcmc.get_samples(group_by_chain=True) #normalized theta
    theta_samples = self.x_to_theta(mcmc.get_samples()) #real theta
    lnP = -mcmc.get_extra_fields()['potential_energy']
    hmc_num_steps = mcmc.get_extra_fields()[
        'num_steps']  # Number of steps in the Hamiltonian trajectory (for diagnostics).
    hmc_tree_depth = np.log2(hmc_num_steps).astype(
        int) + 1  # Tree depth of the Hamiltonian trajectory (for diagnostics).
    hit_max_tree_depth = np.sum(
        hmc_tree_depth == self.mcmc_max_tree_depth)  # Number of transitions that hit the maximum tree depth.
    ms_per_step = 1e3 * total_time / np.sum(hmc_num_steps)

    print(f"*** SUMMARY FOR HMC ***")
    print(f"total_time = {total_time} seconds for the HMC")
    print(f"total_steps = {np.sum(hmc_num_steps)} total steps")
    print(f"ms_per_step = {ms_per_step} ms per step of the HMC")
    print(f"n_eff_mean = {neff_mean} effective sample size, compared to ntot = {self.mcmc_nsteps_tot} total samples.")
    print(f"sec_per_neff = {sec_per_neff:.3f} seconds per effective sample")
    print(f"r_hat_mean = {r_hat_mean}")
    print(f"max_tree_depth encountered = {hmc_tree_depth.max()} in chain")
    print(f"There were {hit_max_tree_depth} transitions that exceeded the max_tree_depth = {self.mcmc_max_tree_depth}")
    print("*************************")

    # Return the values needed
    return x_samples, theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean, \
           hmc_num_steps, hmc_tree_depth, total_time

if out_prefix is not None or debug:
    walkerfile = out_prefix + '_walkers_' +  '.pdf'
    cornerfile = out_prefix + '_corner_' +  '.pdf'
    x_cornerfile = out_prefix + '_x-corner_' +  '.pdf'
    specfile = out_prefix + '_spec_' + '.pdf'
    _x_true = self.x_true[iqso, :] if self.x_true is not None else None
    _theta_true = self.theta_true[iqso, :] if self.theta_true is not None else None
    walker_plot(np.swapaxes(self.x_samples, 0, 1), self.var_label,
                truths=self.x_true[iqso, :] if self.x_true is not None else None,
                walkerfile=walkerfile, linewidth=1.0)
    # Raw x_params corner plot
    corner_plot(x_samples, x_var_label,
                theta_true=_x_true if self.x_true is not None else None,
                cornerfile=x_cornerfile)
    corner_plot(samples, var_label,
                theta_true=self.theta_true if self.theta_true is not None else None,
                cornerfile=cornerfile)
    self.quasar_plot(iqso, self.samples, theta_true=self.theta_true, qso_cont_true=self.qso_cont_true,
                     flux_no_noise=self.flux_no_noise, specfile=specfile)

if out_prefix is not None:
    mcmc_savefile = out_prefix + '.hdf5'
    self.mcmc_save(mcmc_savefile)

def mcmc_save(self, mcmc_savefile):
        """
        Save the MCMC results to an HDF5 file
        Args:
            mcmc_savefile (str):
                output file for MCMC results
        Returns:
        """

    with h5py.File(mcmc_savefile, 'w') as f:
            group = f.create_group('mcmc')
            # Set the attribute parameters of the MCMC
            group.attrs['mcmc_nsteps'] = self.mcmc_nsteps
            group.attrs['mcmc_num_chains'] = self.mcmc_num_chains
            group.attrs['mcmc_warmup'] = self.mcmc_warmup
            group.attrs['mcmc_dense_mass'] = self.mcmc_dense_mass
            group.attrs['mcmc_max_tree_depth'] = self.mcmc_max_tree_depth
            group.attrs['mcmc_init_perturb'] = self.mcmc_init_perturb
            group.attrs['mcmc_nsteps_tot'] = self.mcmc_nsteps_tot
            # Some other parameters of the run
            group.attrs['nqsos'] = self.nqsos
            group.attrs['nspec'] = self.nspec
            group.attrs['ndim'] = self.ndim
            group.attrs['ndim_qso'] = self.ndim_qso
            group.attrs['latent_dim'] = self.QSOmodel.latent_dim
            group.attrs['dv_is_param'] = self.QSOmodel.dv_is_param
            group.attrs['nastro'] = self.nastro
            group.attrs['astro_only'] = self.astro_only
            # MCMC results
            group.create_dataset('neff', data=self.neff)
            group.create_dataset('neff_mean', data=self.neff_mean)
            group.create_dataset('sec_per_neff', data=self.sec_per_neff)
            group.create_dataset('ms_per_step', data=self.ms_per_step)
            group.create_dataset('r_hat', data=self.neff)
            group.create_dataset('r_hat_mean', data=self.neff_mean)
            group.create_dataset('hmc_num_steps', data=self.hmc_num_steps)
            group.create_dataset('hmc_tree_depth', data=self.hmc_num_steps)
            group.create_dataset('runtime', data=self.runtime)
            group.create_dataset('samples', data=self.samples)
            group.create_dataset('x_samples', data=self.x_samples)
            group.create_dataset('ln_probs', data=self.ln_probs)
            # Save the data
            group.create_dataset('wave_rest', data=self.wave_rest)
            group.create_dataset('flux', data=self.flux)
            group.create_dataset('ivar', data=self.ivar)
            group.create_dataset('gpm', data=self.gpm)
            # Some other things related to truths
            if self.theta_true is not None:
                group.create_dataset('ln_prob_true', data=self.ln_prob_true)
            if self.theta_true is not None:
                group.create_dataset('theta_true', data=self.theta_true)
            if self.qso_cont_true is not None:
                group.create_dataset('qso_cont_true', data=self.qso_cont_true)
            if self.flux_no_noise is not None:
                group.create_dataset('flux_no_noise', data=self.flux_no_noise)