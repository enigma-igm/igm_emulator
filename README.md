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

1. **Loads the Training Data**: Loads the noiseless training data at a chosen redshift required for the Lyman-alpha thermal history emulator.
2. **Configures the Emulator**: Finds the best hyperparameters for the emulator by optimizing on validation loss.
3. **Trains the Emulator**: Uses the training data to train the best-configured emulator and gives estimated prediction error.
4. **Visualizes the Training Process**: Shows the learning curve plot; the Emulation vs. Data overplots on training data, testing data; residuals of testing dataset; and estimated NN error matrix with its fraction to the total uncertainties.
5. **Saves the Trained Emulator**: Saves the trained emulator at given redshift for implementation in next step as well as its process plots and prediction error.

#### Running the Script
 
To run the script, use the following command:

```sh
python3 igm_emulator/lya_thermal_emulator_setup.py
```
It will give you a brief greeting and 2 action prompts:

```
***Welcome to the IGM Ly-a thermal Emulator! Please follow the upcoming prompts to guide your actions.***
Optimize hyperparameters of a NN emulator in ~10 min (Y/N)?
Show training of a best-hparam emulator with plots in ~5 sec (Y/N)?
```
> [!CAUTION]
> Please answer Y or N without any spacing or quote.

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

The script will then call `emulator/data_loader.py` to load in the Lyman-alpha auto-correlation functions of the selected dimension (59 bins for **Y** to the second question, 276 bins otherwise) and thermal parameters at the selected redshift for training, validation, and testing. The training will hence start based on the action prompt you chose initially as in the figure, and each emulator training takes about 5 seconds.

<img width="1501" alt="image" src="https://github.com/enigma-igm/igm_emulator/assets/102839205/e4f3b6fa-1f91-4f9f-95ba-a6b251244d94">

It will also give you nice training/performace plots.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/5c6c410e-9f9a-48ad-b955-8d99e81cb741">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/4e6d9e4d-c3f3-440a-b7ca-dc9013dcb601">

And error estimation of the emulator.

<img width="400" alt="image" src="https://github.com/user-attachments/assets/2b317f09-f95c-4e43-84d2-a2a46830d714">
<img width="400" alt="image" src="https://github.com/user-attachments/assets/647f7cf5-8ada-4949-9c0a-8b2fb457269b">


### HMC Inference Setup
The hmc_inference_setup.py script is to implement the IGM emulator for the Hamiltonian Monte Carlo (HMC) thermal parameter inference.

#### Script Description
The `hmc_inference_setup.py` script performs the following tasks:

1. **Configures the HMC Inference Hyperparameters**: Sets up the hyperparameters required for the HMC inference.
2. **Initializes the HMC Inference Model**: Initializes the model used for HMC inference -- we use central models and mocks as examples.
3. **Runs the HMC Inference**: Executes the HMC inference to sample from the posterior distribution.
4. **Saves the Inference Results**: Saves the results of the HMC inference for further analysis.

#### Running the Script
 
To run the script, use the following command:

```sh
python3 igm_emulator/hmc_inference_setup.py
```
This command will again give you a brief greeting and 2 action prompts:
```
***Use the IGM Emulator for thermal parameter inference with HMC!***
Run HMC at central true models/mocks? (Y/N)
Start inference test? (Y/N)
```

- If you want to initialize a model and run the HMC inference for thermal parameters from it, respond with **Y** to the first question. This step takes approximately 10 seconds to infer with 4000 samples.
- Answer **Y** to the second question if you want to test the reliability of the posterior countors from the inference. A **inference test** will be run by repeating a certain number of inference process with different thermal models at given redshift and check if the true credibility level matches with the calculated likelihood.

You can configure the HMC inference by deciding whether use NN error propagted likelihood or not (`Use NN error propagation?`) and the sample size for each chain (`Number of HMC samples (default = 4000)`) by answering 'Y' to either prompt. 

1. Responding with **Y** to the first question will allow you to infer themal parameters at the central model and 2 random central mocks at a given redshift. Both corner plots with parameter posteriors and fit plots of the Lyman-alpha auto-correlation emulation will be saved. Following commands will appear:
  ```
  Use NN error propagation? (Y/N)Y
  Number of HMC samples (default = 4000): 4000
  Starting HMC for the central model and 2 random mocks...

  **Indicate which NN emulator to train/use.**
  ...
  ...
  ```
2. Responding with **Y** to the second question will allow you to infer themal parameters at a number of thermal models (`Number of inference points (default = 100)`) at a given redshift. Use either forward-modeled random mocks or gaussianized mocks to pass to the inference by aswering to `Use forward-modeled mocks? (Y/N)`. A coverage plot will be saved to check if each credibility level matches with our likelihood. Following commands will appear:

  ```
  Use NN error propagation? (Y/N)
  Number of HMC samples (default = 4000): 
  Use forward-modeled mocks? (Y/N)
  Number of inference points (default = 100): 
  Starting inference test...
  
  **Indicate which NN emulator to train/use.**
  ...
  ...
  ```
> [!NOTE]
> The physics of the emulator and data is set up after the HMC setups (`**Indicate which NN emulator to train/use.**`) and is consistent if answer **Y** to both prompts.

In the end, you will have corner plots for parameter inference at cebtral model.
<img width="700" alt="image" src="https://github.com/user-attachments/assets/9e344e38-40d5-4d6e-b7f5-9006246567dd">

And coverage plot for inference test in 100 (default) inferences, shaded region is Poisson error and closer to red line would mark passing the test.
<img width="400" alt="image" src="https://github.com/user-attachments/assets/d450525d-2941-4481-bd81-eb462eef10cd">

#### Results

If you repeat the above procedures for all redshift, you will have the following inference results. 

For noiseless data at central-grid model, dashed line is true values:
<img width="400" alt="image" src="https://github.com/user-attachments/assets/1d22ca3e-069f-4afc-8ea0-c39a114c677e">

For noisy random mocks at central-grid model, dashed line is true values:
<img width="400" alt="image" src="https://github.com/user-attachments/assets/697af92d-1c9f-4f22-b132-699ce67bcca6">


#### Citation
```
@ARTICLE{2024arXiv241006505J,
       author = {{Jin}, Zhenyu and {Wolfson}, Molly and {Hennawi}, Joseph F. and {Gonz{\'a}lez-Hern{\'a}ndez}, Diego},
        title = "{Neural network emulator to constrain the high-$z$ IGM thermal state from Lyman-$\alpha$ forest flux auto-correlation function}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2024,
        month = oct,
          eid = {arXiv:2410.06505},
        pages = {arXiv:2410.06505},
          doi = {10.48550/arXiv.2410.06505},
archivePrefix = {arXiv},
       eprint = {2410.06505},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv241006505J},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
