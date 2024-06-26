import numpy as np
import subprocess
def main():
    print("**Indicate which NN emulator to train/use.**")
    redshift = input("Emulate at Redshift ([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]): ")
    small_scale = input("Less velocity bins of Ly-a forest (Y/N): ") == 'Y'
    n_testing = int(input("Extra testing data number for NN error estimation (<= 1115, default = 0): "))

    # Convert the input to numpy.float64
    try:
        redshift_float64 = np.float64(redshift)
    except ValueError:
        print("Invalid input. Please enter a valid redshift.")
    return redshift_float64, small_scale, n_testing

if __name__ != "__main__":
    redshift, small_scale, n_testing = main()

if __name__ == "__main__":
    print("***Welcome to the IGM Ly-a thermal Emulator! Please follow the upcoming prompts to guide your actions.***")
    emulator_tune = input('Optimize hyperparameters of a NN emulator in ~10 min (Y/N)?') == 'Y'
    emulator_train = input('Show training of a best-hparam emulator with plots in ~5 sec (Y/N)?') == 'Y'

    if emulator_tune:
        print("Running 'python3 hparam_tuning'")
        # Use subprocess to run 'python3 hparam_tuning'
        subprocess.run(['python3', 'emulator/hparam_tuning.py'])

    if emulator_train:
        print("Start re-training emulator...")
        try:
            # Use subprocess to run 'python3 hparam_tuning'
            subprocess.run(['python3', 'emulator/emulator_apply.py'])
        except Exception as e:
            print('Error. Find best-hparam emulator at this redshift first!', e)
