# IGM emulator

This project is designed to set up and train a Neural Network (NN) emulator for the z= 4.5-6 IGM Ly-a thermal parameter inference with Hamiltonian Monte Carlo (HMC).

## Prerequisites

- Ensure you have Python installed on your system. This project is developed using Python.
- Ensure you have access to '/mnt/quasar2/' in the '@igm.physics.ucsb.edu' server to acquire thermal models from **Nyx** simulations that will be read in from 'emulator/data_loader.py'.
- Ensure you have [JAX](https://jax.readthedocs.io/en/latest/installation.html) installed in your environment. This project makes use of the JAX library for automatic differentiation and accelerated linear algebra.

## Setup

1. Clone the repository to your local machine.
2. Navigate to the project directory.
   
## IGM Ly-a Thermal Emulator Setup

Once you have done the prerequisites and setup, it's very easy to train and use an emulator at a given redshift ([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]). The deafault training + validation number of models is 100 with 12 models for testing, while you can select additional testing data for NN emulator error estimation with the setup.
<img width="413" alt="train_sample" src="https://github.com/enigma-igm/igm_emulator/assets/102839205/6700b90f-677e-4750-8a55-4a6d2d49e040">


### Usage

Run the `lya_thermal_emulator_setup.py` script. This script will guide you through the process of setting up and training the emulator.

```bash
python3 lya_thermal_emulator_setup.py
