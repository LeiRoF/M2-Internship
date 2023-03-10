import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration ---------------------------------------------------------------

N = 64 # resolution in pixel
r, dr = np.linspace(-1, 1, N, endpoint=True, retstep=True) # Space range (1D but serve as base for 3D)
T = 10 # Kinetic temperature [K]
CO_fractional_abundance = 1e-4 # particle / hydrogen atom
N2H_fractional_abundance = 1e-7 # particle / hydrogen atom

# Environment -----------------------------------------------------------------

X, Y, Z = np.meshgrid(r, r, r)
R = np.sqrt(X**2 + Y**2 + Z**2)

# Global functions ------------------------------------------------------------

def plummer(r, R=0.75, p=2):
    return 3/(4*np.pi*R**3)*(1 + r**p / R**p)**(-5/2)

# Model cloud description -----------------------------------------------------

def write_LOC_cloud(N, H, fractional_abundance):
    header = np.array([N, N, N,], dtype=np.int32)
    model = np.zeros((N, N, N, 7), dtype=np.float32)

    model[:,:,:,0] = H.density_cube.value # cloud density
    model[:,:,:,1] = T # kinetic temperature
    model[:,:,:,2] = 0 # microturbulence
    model[:,:,:,3] = 0 # macroscopic velocity (x)
    model[:,:,:,4] = 0 # macroscopic velocity (y)
    model[:,:,:,5] = 0 # macroscopic velocity (z)
    model[:,:,:,6] = fractional_abundance # fractional abundance

    with open(f"data/LOC/input_cloud.bin", "wb") as file:
        header.tofile(file)
        model.tofile(file)

# LOC output reading functions ------------------------------------------------

import os, sys
from matplotlib.pylab import *

def LOC_read_spectra_3D(filename):
    """
    Read spectra written by LOC.py (LOC_OT.py; 3D models)
    Usage:
        V, S = LOC_read_spectra_3D(filename)
    Input:
        filename = name of the spectrum file
    Return:
        V  = vector of velocity values, one per channel
        S  = spectra as a cube S[NRA, NDE, NCHN] for NRA
             times NDE points on the sky,
             NCHN is the number of velocity channels
    """
    fp              =  open(filename, 'rb')
    NRA, NDE, NCHN  =  fromfile(fp, int32, 3)
    V0, DV          =  fromfile(fp, float32, 2)
    SPE             =  fromfile(fp, float32).reshape(NRA, NDE,2+NCHN)
    OFF             =  SPE[:,:,0:2].copy()
    SPE             =  SPE[:,:,2:]
    fp.close()
    return V0+arange(NCHN)*DV, SPE
    
    
def LOC_read_Tex_3D(filename):
    """
    Read excitation temperatures written by LOC.py (LOC_OT.py) -- in case of
    a Cartesian grid (no hierarchical discretisation).
    Usage:
        TEX = LOC_read_Tex_3D(filename)
    Input:
        filename = name of the Tex file written by LOC1D.py
    Output:
        TEX = Vector of Tex values [K], one per cell.
    Note:
        In case of octree grids, the returned vector must be
        compared to hierarchy information (e.g. from the density file)
        to know the locations of the cells.
        See the routine OT_GetCoordinatesAllV()
    """
    fp    =  open(filename, 'rb')
    NX, NY, NZ, dummy  =  fromfile(fp, int32, 4)
    TEX                =  fromfile(fp, float32).reshape(NZ, NY, NX)
    fp.close()
    return TEX 

# Initialisation file ---------------------------------------------------------

def write_LOC_config(N, molecule):
    config = f"""\
    cloud          data/LOC/input_cloud.bin       #  cloud defined on a Cartesian grid
    distance       7200.0                         #  cloud distance [pc]
    angle          1.0                            #  model cell size  [arcsec]
    molecule       data/LOC/lamda_database/{molecule}.dat #  name of the Lamda file
    density        1.0                            #  scaling of densities in the cloud file
    temperature    1.0                            #  scaling of Tkin values in the cloud file
    fraction       1.0                            #  scaling o the fraction abundances
    velocity       1.0                            #  scaling of the velocity
    sigma          1.0                            #  scaling o the microturbulence
    isotropic      2.73                           #  Tbg for the background radiation field
    levels         3                              #  number of energy levels in the calculations
    uppermost      3                              #  uppermost level tracked for convergence
    iterations     5                              #  number of iterations
    nside          2                              #  NSIDE parameter determining the number of rays (angles)
    direction      90 0.001                       #  theta, phi [deg] for the direction towards the observer
    points         {N} {N}                        #  number of pixels in the output maps
    grid           1.0                            #  map pixel size [arcsec]
    spectra        1 0  2 1                       #  spectra written for transitions == upper lower (pair of integers)
    transitions    1 0  2 1                       #  Tex saved for transitions (upper and lower levels of each transition)
    bandwidth      6.0                            #  bandwidth in the calculations [km/s]
    channels       128                            #  number of velocity channels
    prefix         data/LOC/output/res            #  prefix for output files
    load           load_level_pop                 #  load level populations
    save           save_level_pop                 #  save level populations
    stop           -1.0                           #  stop if level-populations change below a threshold
    gpu            1                              #  gpu>0 => use the first available gpu
    """

    with open("data/LOC/input_config.ini", "w") as file:
        file.write(config)

# Run LOC ---------------------------------------------------------------------

def run_LOC():

    os.system("python src/LOC/LOC_OT.py data/LOC/input_config.ini")

    os.rename("CO.dump", "data/LOC/output/CO.dump")
    os.rename("N2H+.dump", "data/LOC/output/N2H+.dump")
    os.rename("gauss_py.dat", "data/LOC/output/gauss_py")
    os.rename("save_level_pop", "data/LOC/output/save_level_pop")

# Generated dataset -----------------------------------------------------------

for n_H in np.linspace(1e3, 1e6,10, endpoint=True): # Density from 10^3 to 10^6 hydrogen atom per cm^-3
    for r in np.linspace(0.02, 1.0, 10, endpoint=True): # Core radius from 0.02 to 1 parsec
        for p in np.linspace(1.5, 2.5, 10, endpoint=True): # Sharpness of the plummer profile from 1.5 to 2.5

            profile = plummer(R, r, p)
            density_cube = n_H * profile / np.max(profile)

            # CO simulation
            write_LOC_cloud(N, density_cube, CO_fractional_abundance)
            write_LOC_config("CO", N)
            run_LOC()

            # TODO: Get resulting images

            # N2H+ simulation
            write_LOC_cloud(N, density_cube, N2H_fractional_abundance)
            write_LOC_config("N2H+", N)
            run_LOC()

            # TODO: Get resulting images
            
            # TODO: Run SOC for dust simulation