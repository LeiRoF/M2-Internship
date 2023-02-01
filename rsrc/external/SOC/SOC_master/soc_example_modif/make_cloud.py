from numpy import *
N   =  64                                       # model with N^3 cells
fp  =  open('tmp.cloud', 'w')
asarray([N, N, N, 1, N*N*N], int32).tofile(fp)  #  NX, NY, NZ, LEVELS, CELLS
asarray([N*N*N,], int32).tofile(fp)             #  cells on the first (and only) level
n   = ones((N,N,N), float32)                    #  density cube
n.tofile(fp)
fp.close()
