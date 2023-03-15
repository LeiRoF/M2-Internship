import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import argrelextrema
import subprocess
from LRFutils import progress

# Configuration ---------------------------------------------------------------

N = 64 # resolution in pixel
channels = 128 # number of velocity channels
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

# Gas cloud description -----------------------------------------------------

def write_LOC_cloud(N, density_cube, fractional_abundance):
    header = np.array([N, N, N,], dtype=np.int32)
    model = np.zeros((N, N, N, 7), dtype=np.float32)

    model[:,:,:,0] = density_cube # cloud density
    model[:,:,:,1] = T # kinetic temperature
    model[:,:,:,2] = 0 # microturbulence
    model[:,:,:,3] = 0 # macroscopic velocity (x)
    model[:,:,:,4] = 0 # macroscopic velocity (y)
    model[:,:,:,5] = 0 # macroscopic velocity (z)
    model[:,:,:,6] = fractional_abundance # fractional abundance

    with open(f"data/LOC/input_cloud.bin", "wb") as file:
        header.tofile(file)
        model.tofile(file)

# Dust cloud description -----------------------------------------------------

def write_SOC_cloud(N, density_cube):
    header = np.array([N, N, N, 1, N**3, N**3], dtype=np.int32)
    model = np.zeros((N, N, N, 7), dtype=np.float32)

    model[:,:,:,0] = density_cube # cloud density

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

def write_LOC_config(N, molecule, channels):
    config = \
f"""
cloud          data/LOC/input_cloud.bin       #  cloud defined on a Cartesian grid
molecule       data/LOC/lamda_database/{molecule}.dat #  name of the Lamda file
points         {N} {N}                        #  number of pixels in the output maps

distance       100.0           #  cloud distance [pc]
angle          3.0             #  model cell size  [arcsec]
density        1.0e4           #  scaling of densities in the cloud file
temperature    1.0             #  scaling of Tkin values in the cloud file
fraction       1.0e-4          #  scaling o the fraction abundances
velocity       0.5             #  scaling of the velocity
sigma          0.5             #  scaling o the microturbulence
isotropic      2.73            #  Tbg for the background radiation field
levels         10              #  number of energy levels in the calculations
uppermost      3               #  uppermost level tracked for convergence
iterations     5               #  number of iterations
nside          2               #  NSIDE parameter determining the number of rays (angles)
direction      90 0.001        #  theta, phi [deg] for the direction towards the observer
points         64 64           #  number of pixels in the output maps
grid           3.0             #  map pixel size [arcsec]
spectra        1 0  2 1        #  spectra written for transitions == upper lower (pair of integers)
transitions    1 0  2 1        #  Tex saved for transitions (upper and lower levels of each transition)
bandwidth      1.0             #  bandwidth in the calculations [km/s]
channels       {channels}      #  number of velocity channels
prefix         data/LOC/output/res             #  prefix for output files
load           level_pop.load  #  load level populations
save           level_pop.save  #  save level populations
stop          -1.0             #  stop if level-populations change below a threshold
gpu            1               #  gpu>0 => use the first available gpu
"""

    with open("data/LOC/input_config.ini", "w") as file:
        file.write(config)

# Run LOC ---------------------------------------------------------------------

def run_LOC():

    subprocess.run(["python","src/LOC/LOC_OT.py","data/LOC/input_config.ini"], capture_output=True)

    try:
        os.rename("CO.dump", "data/LOC/output/CO.dump")
    except:
        pass
    try:
        os.rename("N2H+.dump", "data/LOC/output/N2H+.dump")
    except:
        pass
    try:
        os.rename("gauss_py.dat", "data/LOC/output/gauss_py")
    except:
        pass
    try:
        os.rename("level_pop", "data/LOC/output/level_pop.save")
    except:
        pass

# Generate graphs -------------------------------------------------------------

def generate_graphs(v, sepctra, molecule):
    
    # Average spectra oveers the whole picture
    avg_spectra = np.sum(spectra, axis=(0,1))

    plt.figure()
    plt.plot(v, avg_spectra)
    plt.xlabel("Velocity [km/s]")
    plt.ylabel("Intensity [K km/s]")
    plt.title("Average spectra")
    plt.savefig(f"data/LOC/output/{molecule}_avg_spectra.png")

    maximums = argrelextrema(avg_spectra, np.greater)
    print("Maximums found at: ", v[maximums], "Hz")

    mid = len(v)//2
    observation = spectra[:,:,mid]
    print(f"Observation 1 of {molecule}, made at: ", v[mid], "Hz")
    plt.figure()
    plt.imshow(observation)
    plt.colorbar()
    plt.title(f"Observation at {v[mid]} Hz")
    plt.savefig(f"data/LOC/output/{molecule}_observation_1_at_{v[mid]}Hz.png")

    for i, max in enumerate(maximums[0]):
        print(f"Observation {i+2} of {molecule}, made at: ", v[max], "Hz")
        plt.figure()
        plt.imshow(spectra[:,:,max])
        plt.colorbar()
        plt.title(f"Observation at {v[max]} Hz")
        plt.savefig(f"data/LOC/output/{molecule}_observation_{i+2}_at_{v[max]}Hz.png")

    mosaic = 10
    fig, axs = plt.subplots(mosaic, mosaic, figsize=(30,30))
    for i in range(mosaic):
        for j in range(mosaic):
            # Plotting spectra for 10 equally distant pixels in each row and column
            Nx, Ny, Nf = spectra.shape
            axs[i,j].plot(v, spectra[Nx//mosaic*i, Ny//mosaic*j, :])
    
    fig.savefig(f"data/LOC/output/{molecule}_spectra_mosaic.png")

    plt.figure()
    plt.imshow(np.sum(spectra, axis=-1))
    plt.colorbar()
    plt.title("Line area")
    plt.savefig(f"data/LOC/output/{molecule}_line_area.png")

# Generated dataset -----------------------------------------------------------

test = False

n_List = np.linspace(1e3,  1e6, 2, endpoint=True) # Density from 10^3 to 10^6 hydrogen atom per cm^-3
r_List = np.linspace(0.02, 1.0, 10, endpoint=True) # Core radius from 0.02 to 1 parsec
p_List = np.linspace(1.5,  2.5, 2, endpoint=True) # Sharpness of the plummer profile from 1.5 to 2.5

CO_avg_spectrum_datacube = np.zeros((10, 10, 10, channels))
N2H_avg_spectrum_datacube = np.zeros((10, 10, 10, channels))

bar = progress.Bar(len(n_List) * len(r_List) * len(p_List), prefix="Generating images")

for i, n_H in enumerate(n_List): 
    for j, r in enumerate(r_List): 
        for k, p in enumerate(p_List): 

            bar(i*len(r_List)*len(p_List) + j*len(p_List) + k)

            if test:
                n_H = 1e3
                r = 0.02
                p = 1.5

            # Generate cloud --------------------------------------------------

            profile = plummer(R, r, p)
            density_cube = n_H * profile / np.max(profile)

            # CO simulation ---------------------------------------------------

            write_LOC_cloud(N, density_cube, CO_fractional_abundance)
            write_LOC_config(N, "CO", channels)
            run_LOC()

            v, spectra = LOC_read_spectra_3D("data/LOC/output/res_CO_01-00.spe")

            CO_avg_spectrum_datacube[i,j,k,:] = np.sum(spectra, axis=(0,1))

            if test:
                generate_graphs(v, spectra, "CO")
            
            # N2H+ simulation -------------------------------------------------

            write_LOC_cloud(N, density_cube, N2H_fractional_abundance)
            write_LOC_config(N, "N2H+", channels)
            run_LOC()

            v, spectra = LOC_read_spectra_3D("data/LOC/output/res_N2H+_01-00.spe")

            N2H_avg_spectrum_datacube[i,j,k,:] = np.sum(spectra, axis=(0,1))

            if test:
                generate_graphs(v, spectra, "CO")

            # Dust simulation -------------------------------------------------
            
            # TODO: Run SOC for dust simulation

            # Compute only the first loop if it's in a test mode
            if test:
                break
        if test:
            break
    if test:
        break

bar(len(n_List) * len(r_List) * len(p_List))

np.savez_compressed("data/spectra_datacubes",
                    CO = CO_avg_spectrum_datacube,
                    N2H = N2H_avg_spectrum_datacube,
                    v = v,
                    n_H = n_List,
                    r = r_List,
                    p = p_List,
                )


            

            