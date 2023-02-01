import sys
sys.path.append('/home/rabnaeb/Documents/COURS/Master2/stage/SOC/')
from SOC_aux import *

F = logspace(log10(um2f(3000.0)), log10(um2f(0.1)), 55)

REQ = [70.0, 160.0, 250.0, 350.0, 500.0, 1000.0, 2000.0]
for x in REQ:
    d = abs(x-f2um(F))
    i = argmin(d)
    if (min(d)<0.1): continue
    elif (min(d)<0.001*F[i]):
        F[i] = um2f(x)
    else:
        if (F[i]<um2f(x)): i+= 1
        F = concatenate((F[0:(i-1)], [um2f(x),], F[i:]))

fp = open('freq.dat', 'w')
for f in F:
    fp.write('%12.4e\n' % f)
fp.close()