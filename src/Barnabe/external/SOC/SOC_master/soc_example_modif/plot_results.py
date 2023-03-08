from matplotlib.pylab import *

freq     = loadtxt('freq.dat')            # read the frequencies
nfreq    = len(freq)                      # ... number of frequencies
figure(1, figsize=(8,3))
"""
fp       = open('map_dir_00.bin', 'rb')   # open the surface-brightness file
dims     = fromfile(fp, int32, 2)         # read map dimensions in pixels
S        = fromfile(fp, float32).reshape(nfreq, dims[0], dims[1]) # read intensities
S       *= 1.0e-5                         # convert surface brightness from Jy/sr to MJy/sr
fp.close()

ifreq    = argmin(abs(freq-2.9979e8/250.0e-6)) # Choose frequency closest to 250um

ax = subplot(121)
imshow(S[ifreq,:,:])                      # plot the 250um surface brightness map
title("Surface brightness")
colorbar()
text(1.34, 0.5, r'$S_{\nu} \/ \/ \rm (MJy \/ sr^{-1})$', transform=ax.transAxes,
va='center', rotation=90)
"""
fp       = open('tmp.T', 'rb')            # open the file with dust temperatures
NX, NY, NZ, LEVELS, CELLS, CELLS_LEVEL_1 = fromfile(fp, int32, 6)
T        = fromfile(fp, float32).reshape(NZ, NY, NX) # T is simply a 64x64x64 cell cube

ax = subplot(122)
imshow(T[NZ//2, :, :])                    # plot cross section through the model centre
title("Dust temperature")
colorbar()
text(1.34, 0.5, r'$T_{\rm dust} \/ \/ \rm (K)$', transform=ax.transAxes, va='center', rotation=90)

show()

