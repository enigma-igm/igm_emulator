from os import path
from setuptools import find_packages, setup

if path.exists('README.md'):
    with open('README.md') as readme:
        long_description = readme.read()



setup(
    name="igm_emulator",
    version='0.0.dev0',
    description="NN emulator on lyman-alpha auto-correlation function.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/enigma-igm/igm_emulator",
    author='Linda Jin and friends',
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "numpy",
        "scipy",
        "tqdm",
        "scikit-learn",
        "matplotlib",
        "jaxlib",
        "jax",
        "flax",
        "optax",
        "clu",
        "numpyro",
        "ipython",
        "pandas",
        "optuna",
        "plotly",
        "corner",
        "h5py",
        "arviz"
    ], dependency_links=[],
    scripts=[],)

