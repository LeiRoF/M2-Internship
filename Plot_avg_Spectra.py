import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider

data = np.load("data/spectra_datacube.npz")

CO = data["CO"]
v = data["v"]
n_H = data["n_H"]
r = data["r"]
p = data["p"]

fig = plt.figure(figsize=(10,10))
matplotlib.rcParams.update({'font.size': 22})

###############################################################################
# INTERACTIVE PLOT
###############################################################################

# Create main axis
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=0.2, top=0.75)

# Create axes for sliders
ax_amp_X = fig.add_axes([0.3, 0.95, 0.4, 0.05])
ax_amp_X.spines['top'].set_visible(True)
ax_amp_X.spines['right'].set_visible(True)

ax_amp_Y = fig.add_axes([0.3, 0.90, 0.4, 0.05])
ax_amp_Y.spines['top'].set_visible(True)
ax_amp_Y.spines['right'].set_visible(True)

ax_amp_Z = fig.add_axes([0.3, 0.85, 0.4, 0.05])
ax_amp_Z.spines['top'].set_visible(True)
ax_amp_Z.spines['right'].set_visible(True)

ax_det = fig.add_axes([0.3, 0.80, 0.4, 0.05])
ax_det.spines['top'].set_visible(True)
ax_det.spines['right'].set_visible(True)

ax_phi = fig.add_axes([0.3, 0.75, 0.4, 0.05])
ax_phi.spines['top'].set_visible(True)
ax_phi.spines['right'].set_visible(True)

# Create sliders
slider_amp_X = Slider(ax=ax_amp_X, label=r'H density [$molecule/cm^3$]', valmin=n_H[0], valmax=n_H[-1],
              valinit=1, valstep=n_H[1]-n_H[0], valfmt=' %.3f', facecolor='#cc7000')

slider_amp_Y = Slider(ax=ax_amp_Y, label=r'Core radius [$pc$]', valmin=r[0], valmax=r[-1], 
             valinit=1, valstep=r[1]-r[0], valfmt='%.3f', facecolor='#cc7000')

slider_amp_Z = Slider(ax=ax_amp_Z, label='Plummer sharpness', valmin=p[0], valmax=p[-1], 
             valinit=1, valstep=p[1]-p[0], valfmt='%.3f', facecolor='#cc7000')

# Plot default data
im = ax.plot(v, CO[n_H[0], r[0], p[0], :] cmap="inferno")

# Update values
def update(val):
    nH_val = slider_amp_X.val
    r_val = slider_amp_Y.val
    p_val = slider_amp_Z.val

    im.plot(v, CO[np.where(n_H==nH_val), np.where(r==r_val), np.where(p==p_val), :], color="white", linewidth=2)
    fig.canvas.draw_idle()

slider_amp_X.on_changed(update)
slider_amp_Y.on_changed(update)
slider_amp_Z.on_changed(update)

plt.show()


