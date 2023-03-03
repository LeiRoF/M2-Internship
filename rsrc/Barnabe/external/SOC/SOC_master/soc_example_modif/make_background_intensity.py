import sys
sys.path.append('/home/rabnaeb/Documents/COURS/Master2/stage/SOC/')
from SOC_aux import *

test = fromfile('bg_intensity.bin')
savetxt('bg_intensity.dat', test)

freq = loadtxt('freq.dat')
d = loadtxt('isrf3.dat')
ip = interp1d(d[:, 0], d[:, 1])
I = ip(freq)
asarray(I, float32).tofile('bg_intensity.dat')