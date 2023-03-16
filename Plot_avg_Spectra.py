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
matplotlib.rcParams.update({'font.size': 22})

###############################################################################
# INTERACTIVE PLOT
###############################################################################

# Create main axis
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=0.2, top=0.75)

# Create axes for sliders
ax_amp_X = fig.add_axes([0.3, 0.90, 0.4, 0.05])
ax_amp_X.spines['top'].set_visible(True)
ax_amp_X.spines['right'].set_visible(True)

ax_amp_Y = fig.add_axes([0.3, 0.85, 0.4, 0.05])
ax_amp_Y.spines['top'].set_visible(True)
ax_amp_Y.spines['right'].set_visible(True)

ax_amp_Z = fig.add_axes([0.3, 0.80, 0.4, 0.05])
ax_amp_Z.spines['top'].set_visible(True)
ax_amp_Z.spines['right'].set_visible(True)

# Create sliders
slider_amp_X = Slider(ax=ax_amp_X, label=r'$\rho$ [#/cm^3]', valmin=n_H[0], valmax=n_H[-1],
              valinit=1, valstep=n_H[1]-n_H[0], valfmt=' %.3f', facecolor='#cc7000')

slider_amp_Y = Slider(ax=ax_amp_Y, label=r'$r$ [pc]', valmin=r[0], valmax=r[-1], 
             valinit=1, valstep=r[1]-r[0], valfmt='%.3f', facecolor='#cc7000')

slider_amp_Z = Slider(ax=ax_amp_Z, label='Sharpness', valmin=p[0], valmax=p[-1], 
             valinit=1, valstep=p[1]-p[0], valfmt='%.3f', facecolor='#cc7000')

# Plot default data
line, = ax.plot(v, CO[0, 0, 0, :])
ax.set_ylim(0, CO.max())
ax.grid()
ax.set_xlabel(r'Velocity [km/s]')
ax.set_ylabel(r'Intensity [K]')
ax.set_title(r'CO(1-0)')

# Update values
def update(val):
    nH_val = slider_amp_X.val
    r_val = slider_amp_Y.val
    p_val = slider_amp_Z.val

    data = CO[np.where(n_H==nH_val), np.where(r==r_val), np.where(p==p_val), :].flatten()

    line.set_ydata(data)
    fig.canvas.draw_idle()

slider_amp_X.on_changed(update)
slider_amp_Y.on_changed(update)
slider_amp_Z.on_changed(update)

plt.show()


