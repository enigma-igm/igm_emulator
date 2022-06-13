
import os
import numpy as np
from matplotlib import pyplot as plt
from atropy.table import Table
from igm_emulator.emulator.utils import lhs

# Routine to randomly subsample a set of correlation function simulations

corr_table = Table.read('/Users/mwolfson/data/corr_table.fits')

seed =12345
# random number generator
rng = np.random.default_rng(seed)
