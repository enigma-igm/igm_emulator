from os import path
from setuptools import find_packages, setup

if path.exists('README.md'):
    with open('README.md') as readme:
        long_description = readme.read()



setup(
    name="igm_emulator",
    version='0.0.dev0',
    description="Determining the best description of quasar spectrum",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/enigma-igm/igm_emulator",
    author='Linda Jin and friends',
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "numpy",
        "astropy",
        "matplotlib",
    ], dependency_links=[],
    scripts=[],)

