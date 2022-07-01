import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm
from igm_emulator.emulator.utils import *
from igm_emulator.scripts.grab_models import final_samples

H=final_samples
#H= norm(loc=0, scale=1).ppf(lhd)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(H[:,0],H[:,1],H[:,2],c=H[:,2], cmap='viridis', linewidth=0.5)
plt.show()