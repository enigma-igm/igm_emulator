import numpy as np
import subprocess
from igm_emulator.hmc.inference_test import INFERENCE_TEST

def main():
    redshift = input("Emulate at Redshift ([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]): ")
    small_scale = input("Less velocity bins of Ly-a forest ([True, False]): ")
    n_testing = int(input("Extra testing data number: "))

    # Convert the input to numpy.float64
    try:
        redshift_float64 = np.float64(redshift)
    except ValueError:
        print("Invalid input. Please enter a valid redshift.")
    return redshift_float64, small_scale, n_testing

redshift, small_scale, n_testing = main()

if __name__ == "__main__":
    emulator_train = input('Re-train the best-hparam emulator with plots (Y/N)?') == 'Y'
    inf_model_values = input('Run HMC at central models/mocks (Y/N)?') == 'Y'
    inf_test_run = input('Start inference test (Y/N)?') == 'Y'

    if emulator_train:
        print(f"Start re-training emulator at redshift {redshift}...")
        # Use subprocess to run 'python3 hparam_tuning'
        subprocess.run(['python3', 'emulator/emulator_apply.py'])

    if inf_model_values:
        print(f"Start HMC at central models at redshift {redshift}...")
        # Use subprocess to run 'python3 hmc_run.py'
        subprocess.run(['python3', 'hmc/hmc_run.py'])

    if inf_test_run:
        print(f"Start inference test at redshift {redshift}...')
        nn_err_prop = input('Use NN error propagation (Y/N)?') == 'Y'
        forward_mocks = input('Use forward mocks (Y/N)?') == 'N'
        try:
            hmc_infer = INFERENCE_TEST(gaussian_bool=forward_mocks, ngp_bool=False, emu_test_bool=False, nn_err_prop_bool=nn_err_prop,
                                       n_inference=100)


            hmc_infer.mocks_sampling()
            hmc_infer.inference_test_run()
            hmc_infer.coverage_plot()
        except Exception as e:
            print('Error. Train emulator at this redshift first!', e)
            print("Running 'python3 hparam_tuning'")

            # Use subprocess to run 'python3 hparam_tuning'
            subprocess.run(['python3', 'emulator/hparam_tuning.py'])
