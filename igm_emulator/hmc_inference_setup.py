
def run_hmc(num_samples, nn_err_prop_bool):
    print("Starting HMC for the central model and 2 random mocks...")
    from igm_emulator.hmc.hmc_run import run_central_HMC
    run_central_HMC(num_samples, nn_err_prop_bool)

def start_inference_test(num_samples, nn_err_prop, forward_mocks, num_inference):
    print("Starting inference test...")
    from igm_emulator.hmc.inference_test import INFERENCE_TEST
    hmc_infer = INFERENCE_TEST(gaussian_bool=forward_mocks, ngp_bool=False, emu_test_bool=False,
                               nn_err_prop_bool=nn_err_prop, n_inference=num_inference, num_hmc_samples=num_samples)
    hmc_infer.mocks_sampling()
    hmc_infer.inference_test_run()
    hmc_infer.coverage_plot()

def main():
    print("***Use the IGM Emulator for thermal parameter inference with HMC!***")
    print('Here you can: 1.run parameter inference with central true models/mocks, 2.start inference test at a redshift.')
    if input('1. Run HMC at central true models/mocks? (Y/N)') == 'Y':
        nn_err_prop = input('Use NN error propagation? (Y/N)') == 'Y'
        num_samples = int(input('Number of HMC samples (default = 4000): '))
        try:
            run_hmc(num_samples, nn_err_prop)
        except Exception as e:
            print('Error. Train emulator first!', e)

    if input('2. Start inference test? (Y/N)') == 'Y':
        try:
            nn_err_prop
        except NameError:
            nn_err_prop = input('Use NN error propagation? (Y/N)') == 'Y'
        try:
            num_samples
        except NameError:
            num_samples = int(input('Number of HMC samples (default = 4000): '))
        forward_mocks = input('Use forward-modeled mocks? (Y/N)') == 'N'
        if forward_mocks:
            print('Using Gaussianized mocks...')
        num_inference = int(input('Number of inference points (default = 100): '))
        try:
            start_inference_test(num_samples, nn_err_prop, forward_mocks, num_inference)
        except Exception as e:
            print('Error. Train emulator first!', e)

if __name__ == "__main__":
    main()