from igm_emulator.hmc.inference_test import INFERENCE_TEST
import subprocess

print("***Use the IGM Emulator for thermal parameter inference! Please follow the upcoming prompts to guide your actions.***")
inf_model_values = input('Run HMC at central true models/mocks (Y/N)?') == 'Y'
inf_test_run = input('Start inference test (Y/N)?') == 'Y'

if inf_model_values:
    print(f"Start HMC at central models at redshift {redshift}...")
    # Use subprocess to run 'python3 hmc_run.py'
    subprocess.run(['python3', 'hmc/hmc_run.py'])

if inf_test_run:
    print(f"Start inference test at redshift {redshift}...")
    nn_err_prop = input('Use NN error propagation (Y/N)?') == 'Y'
    forward_mocks = input('Use forward mocks (Y/N)?') == 'N'
    try:
        hmc_infer = INFERENCE_TEST(gaussian_bool=forward_mocks, ngp_bool=False, emu_test_bool=False,
                                   nn_err_prop_bool=nn_err_prop,
                                   n_inference=100)

        hmc_infer.mocks_sampling()
        hmc_infer.inference_test_run()
        hmc_infer.coverage_plot()
    except Exception as e:
        print('Error. Train emulator at this redshift first!', e)
        print("Running 'python3 hparam_tuning'")

        # Use subprocess to run 'python3 hparam_tuning'
        subprocess.run(['python3', 'emulator/hparam_tuning.py'])
