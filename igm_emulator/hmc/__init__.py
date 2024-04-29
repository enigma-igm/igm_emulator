"""hmc - thermal parameter inference with the Hamiltonian Monte Carlo method.

It provides

HMC with Neural Network: hmc_nn_inference,
HMC with nearest-neighbor: hmc_ngp_inference.

for inference the thermal parameters of the IGM.

It also runs a statstical inference test: inference_test.
It's backed by the NumPyro library and the HMC framework.
"""