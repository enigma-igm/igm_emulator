# IGM emulator

This project is designed to train and apply a Neural Network (NN) emulator for using the IGM Lyman-alpha auto-correlation functions at $z = 4.5-6.0$ to infer the thermal history with Hamiltonian Monte Carlo (HMC).

## Table of Contents

- [Setup](#Setup)
- [Usage](#Usage)
  - [Lyman-alpha Thermal Emulator Setup](#lyman-alpha-thermal-emulator-setup)
  - [HMC Inference Setup](#hmc-inference-setup)
- [Contributing](#contributing)
- [License](#license)

## Setup

- Ensure you have Python installed on your system. This project is developed using Python.
- Ensure you have access to `/mnt/quasar2/` in the `@igm.physics.ucsb.edu` server to acquire thermal models from **Nyx** simulations that will be read in from 'emulator/data_loader.py'.
- Ensure you have [JAX](https://jax.readthedocs.io/en/latest/installation.html) installed in your environment. This project makes use of the JAX library for automatic differentiation and accelerated linear algebra.

Clone the repository:

```sh
git clone https://github.com/enigma-igm/igm_emulator.git
cd igm_emulator
```

## Usage

### Lyman-alpha Thermal Emulator Setup

Once you have done the prerequisites and setup, it's very easy to train an emulator at any redshift ([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]). The deafault **training + validation** number of models is 100 with 12 models for **testing**, while you can select additional testing data for NN emulator error estimation with the setup.

<img width="413" alt="train_sample" src="https://github.com/enigma-igm/igm_emulator/assets/102839205/6700b90f-677e-4750-8a55-4a6d2d49e040">

#### Script Description

The `lya_thermal_emulator_setup.py` script performs the following tasks:

1. **Loads the Training Data**: Loads the training data at a chosen redshift required for the Lyman-alpha thermal history emulator.
2. **Configures the Emulator**: Finds the best hyperparameters for the emulator by optimizing on validation loss.
3. **Trains the Emulator**: Uses the training data to train the best-configured emulator and gives estimated prediction error.
4. **Visualizes the Training Process**: Shows the learning curve plot; the Emulation vs. Data overplots on training data, testing data; residuals of testing dataset; and estimated NN error matrix with its fraction to the total uncertainties.
5. **Saves the Trained Emulator**: Saves the trained emulator at given redshift for implementation in next step as well as its process plots and prediction error.

#### Running the Script
 
To run the script, use the following command:

```sh
python3 igm_emulator/lya_thermal_emulator_setup.py
```
It will give you a brief greeting and 2 prompts:

```
***Welcome to the IGM Ly-a thermal Emulator! Please follow the upcoming prompts to guide your actions.***

Optimize hyperparameters of a NN emulator in ~10 min (Y/N)?

Show training of a best-hparam emulator with plots in ~5 sec (Y/N)?
```
> [!CAUTION]
> Please answer Y or N without any spacing or quotes.

- If you want to **tune the hyperparameters** of the emulator, respond with **Y** to the first question. This step takes approximately 10 minutes to experiment with 100 configurations.
- After tuning the hyperparameters, you can **save the training results and plots** with the optimal-configured emulator by responding with **Y** to the second question. This step takes approximately 5 seconds for 2000 epochs of training.
- If you have **already tuned the hyperparameters** and want to revisit the training process with additional diagnostic plots or additional points for error estimation, respond with **N** to the first prompt and **Y** to the second prompt.

When you answer 'Y' to either one of the prompt, you will be directed to select the physics of the emulator (i.e. what redshift of thermal history you want to emulate on, what dimension of Lyman-alpha auto-correlation functions you want to train on) and also select additional testing data for NN emulator error estimation. Here's an example for redshift $z=5.7$

```
**Indicate which NN emulator to train/use.**
Emulate at Redshift ([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]): 5.7 
Less velocity bins of Ly-a forest (Y/N): Y
Extra testing data number for NN error estimation (<= 1115, default = 0): 0
Loading parameter grid for redshift 5.7...
```

The script will then call `emulator/data_loader.py` to load in the Lyman-alpha auto-correlation functions of the selected dimension (59 bins for **Y** to the second question, 276 bins otherwise) and thermal parameters at the selected redshift for training, validation, and testing. The training will hence start based on the action prompt you chose initially. 

### HMC Inference Setup
The hmc_inference_setup.py script is used to set up the Hamiltonian Monte Carlo (HMC) inference for the implementation of the IGM emulator.

#### Script Description
The `hmc_inference_setup.py` script performs the following tasks:

1. **Configures the HMC Inference Hyperparameters**: Sets up the hyperparameters required for the HMC inference.
2. **Initializes the HMC Inference Model**: Initializes the model used for HMC inference.
3. **Runs the HMC Inference**: Executes the HMC inference to sample from the posterior distribution.
4. **Saves the Inference Results**: Saves the results of the HMC inference for further analysis.

