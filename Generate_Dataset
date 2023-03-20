import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *
from scipy.signal import argrelextrema
import subprocess
from LRFutils import progress



#==============================================================================
# INIT
#==============================================================================



# Configuration ---------------------------------------------------------------

N = 64 # resolution in pixel
r, dr = np.linspace(-1, 1, N, endpoint=True, retstep=True) # Space range (1D but serve as base for 3D)

channels = 128 # number of velocity channels

T = 10 # Kinetic temperature [K]

CO_fractional_abundance  = 1e-4 # particle / hydrogen atom
N2H_fractional_abundance = 1e-7 # particle / hydrogen atom

n_List = np.linspace(1e3,  1e6, 10, endpoint=True) # Density from 10^3 to 10^6 hydrogen atom per cm^-3
r_List = np.linspace(0.02, 1.0, 10, endpoint=True) # Core radius from 0.02 to 1 parsec
p_List = np.linspace(1.5,  2.5, 10, endpoint=True) # Sharpness of the plummer profile from 1.5 to 2.5


# Environment -----------------------------------------------------------------

X, Y, Z = np.meshgrid(r, r, r)
R = np.sqrt(X**2 + Y**2 + Z**2)

# Global functions ------------------------------------------------------------

def plummer(r:float, R:float, p:float) -> float:

    return 3/(4 * np.pi * R**3) * (1 + r**p / R**p)**(-5/2)



#==============================================================================
# LOC
#==============================================================================



# Gas cloud description -------------------------------------------------------

def write_LOC_cloud(N:int, density_cube:ndarray[float], fractional_abundance:float):

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

# Initialisation file ---------------------------------------------------------

def write_LOC_config(N:int, molecule:str, channels:int):

    config = f"""
        cloud          data/LOC/input_cloud.bin               #  cloud defined on a Cartesian grid
        molecule       data/LOC/lamda_database/{molecule}.dat #  name of the Lamda file

        points         {N} {N}             #  number of pixels in the output maps
        distance       100.0               #  cloud distance [pc]
        angle          3.0                 #  model cell size  [arcsec]
        density        1.0e4               #  scaling of densities in the cloud file
        temperature    1.0                 #  scaling of Tkin values in the cloud file
        fraction       1.0e-4              #  scaling o the fraction abundances
        velocity       0.5                 #  scaling of the velocity
        sigma          0.5                 #  scaling o the microturbulence
        isotropic      2.73                #  Tbg for the background radiation field
        levels         10                  #  number of energy levels in the calculations
        uppermost      3                   #  uppermost level tracked for convergence
        iterations     5                   #  number of iterations
        nside          2                   #  NSIDE parameter determining the number of rays (angles)
        direction      90 0.001            #  theta, phi [deg] for the direction towards the observer
        points         64 64               #  number of pixels in the output maps
        grid           3.0                 #  map pixel size [arcsec]
        spectra        1 0  2 1            #  spectra written for transitions == upper lower (pair of integers)
        transitions    1 0  2 1            #  Tex saved for transitions (upper and lower levels of each transition)
        bandwidth      1.0                 #  bandwidth in the calculations [km/s]
        channels       {channels}          #  number of velocity channels
        prefix         data/LOC/output/res #  prefix for output files
        load           level_pop.load      #  load level populations
        save           level_pop.save      #  save level populations
        stop          -1.0                 #  stop if level-populations change below a threshold
        gpu            1                   #  gpu>0 => use the first available gpu
        """.replace("\n        ","\n")

    with open("data/LOC/input_config.ini", "w") as file:
        file.write(config)

# LOC output reading functions ------------------------------------------------

def LOC_read_spectra_3D(filename:str) -> tuple[ndarray,ndarray]:

    with open(filename, 'rb') as fp:
        NRA, NDE, NCHN  =  fromfile(fp, int32, 3)
        V0, DV          =  fromfile(fp, float32, 2)
        SPE             =  fromfile(fp, float32).reshape(NRA, NDE,2+NCHN)
        OFF             =  SPE[:,:,0:2].copy()
        SPE             =  SPE[:,:,2:]

    return V0+arange(NCHN)*DV, SPE
    
    
def LOC_read_Tex_3D(filename):

    fp    =  open(filename, 'rb')
    NX, NY, NZ, dummy  =  fromfile(fp, int32, 4)
    TEX                =  fromfile(fp, float32).reshape(NZ, NY, NX)
    fp.close()
    return TEX 

# Run LOC ---------------------------------------------------------------------

def run_LOC():

    subprocess.run(["python","src/LOC/LOC_OT.py","data/LOC/input_config.ini"], capture_output=True)

    def move_file(src, dst):
        try:
            os.rename(src, dst)
        except:
            pass

    move_file("CO.dump", "data/LOC/output/CO.dump")
    move_file("N2H+.dump", "data/LOC/output/N2H+.dump")
    move_file("gauss_py.dat", "data/LOC/output/gauss_py")
    move_file("level_pop", "data/LOC/output/level_pop.save")



#==============================================================================
# SOC
#==============================================================================



# Dust cloud description -----------------------------------------------------

def write_SOC_cloud(N:int, density_cube:ndarray[float]):

    header = np.array([N, N, N, 1, N**3, N**3], dtype=np.int32)
    model = np.zeros((N, N, N, 7), dtype=np.float32)

    model[:,:,:,0] = density_cube # cloud density

    with open(f"data/SOC/input_cloud.bin", "wb") as file:
        header.tofile(file)
        model.tofile(file)

# Initialisation file ---------------------------------------------------------

def write_SOC_config(N:int):

    config = f"""
        gridlength      0.01                      # root grid cells have a size of 0.01 pc each
        cloud           data/SOC/input_cloud.bin  # density field (reference to a file)
        mapping         {N} {N} 1.0               # output 64x64 pixels, pixel size == root-grid cell size
        density         1.0e4                     # scale values read from tmp.cloud
        seed           -1.0                       # random seed for random numbers
        directions      0.0  0.0                  # observer in direction (theta,phi)
        optical         data/SOC/aSilx.dust       # dust optical parameters
        dsc             data/SOC/aSilx.dsc  2500  # dust scattering function
        bgpackets       999999                    # photon packages simulated from the background
        background      data/SOC/bg_intensity.bin # background intensity at the
        iterations      1                         # one iteration is enough
        prefix          data/LOC/output/res       # prefix for output files
        absorbed        absorbed.save             # save absorptions to a file
        emitted         emitted.save              # save dust emission to a file
        noabsorbed                                # actually, we integrate absorptions on the fly and skip the *file*
        temperature     temperature.save          # save dust temperatures
        device          g                         # run calculations on a GPU
        CLT                                       # temperature caclulations done on the device
        CLE                                       # emission calculations done on the device
        """.replace("\n        ","\n")

    with open("data/SOC/input_config.ini", "w") as file:
        file.write(config)

# Run SOC ---------------------------------------------------------------------

def run_SOC():

    os.system("python src/SOC/ASOC.py data/SOC/input_config.ini")

    def move_file(src, dst):
        try:
            os.rename(src, dst)
        except:
            pass

    # TODO: move output files



#==============================================================================
# GENERATE DATASET
#==============================================================================


bar = progress.Bar(len(n_List) * len(r_List) * len(p_List), prefix="Generating images")

for i, n_H in enumerate(n_List): 
    for j, r in enumerate(r_List): 
        for k, p in enumerate(p_List): 

            # Updating progress bar
            bar(i*len(r_List)*len(p_List) + j*len(p_List) + k)

            # Generate cloud --------------------------------------------------

            profile = plummer(R, r, p)
            density_cube = n_H * profile / np.max(profile)

            # # CO simulation ---------------------------------------------------

            # write_LOC_cloud(N, density_cube, CO_fractional_abundance)
            # write_LOC_config(N, "CO", channels)
            # run_LOC()

            # CO_v, CO_cube = LOC_read_spectra_3D("data/LOC/output/res_CO_01-00.spe")
            
            # # N2H+ simulation -------------------------------------------------

            # write_LOC_cloud(N, density_cube, N2H_fractional_abundance)
            # write_LOC_config(N, "N2H+", channels)
            # run_LOC()

            # N2H_v, N2H_cube = LOC_read_spectra_3D("data/LOC/output/res_N2H+_01-00.spe")

            # Dust simulation -------------------------------------------------
            
            write_SOC_cloud(N, density_cube)
            write_SOC_config(N)
            run_SOC()

            # Save data -------------------------------------------------------

            np.savez_compressed(f"data/dataset/n={n_H:.2f}_r={r:.2f}_p={p:.2f}.npz",
                # CO_cube = CO_cube,
                # N2H_cube = N2H_cube,
                # CO_v = CO_v,
                # N2H_v = N2H_v,
                density_cube = density_cube,
                n_H = n_H,
                r = r,
                p = p,
            )

            break
        break
    break

# End progress bar
bar(len(n_List) * len(r_List) * len(p_List))


            

            