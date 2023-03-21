import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider

data = np.load("data/spectra_datacubes.npz")

v = data["v"]
CO = data["CO"] / len(v)
n_H = data["n_H"]
r = data["r"]
p = data["p"]

fig = plt.figure(figsize=(10,7))

# Create main axis
ax = fig.add_subplot(111)

# Plot default data

for i, R in enumerate(r):
    line, = ax.plot(v, CO[1, i, 1, :], label=r"$r=" + f"{R:.3f}" + r"$ pc")
ax.set_ylim(0, CO.max())
ax.grid()
ax.legend()
ax.set_xlabel(r'Velocity [km/s]')
ax.set_ylabel(r'Intensity [K]')
ax.set_title(r'CO(1-0)')

plt.show()


