from functions import *
import os
from amuse.units import constants

#Constants
kb = constants.kB
mp = constants.proton_mass
G = constants.G

# Load parameters
p = simuini()

# Thermal to potential energy ratio
ethep = (5.0 * p['radius'] * kb * p['T']) / (2.0 * G * p['mass'] * p['mu'] * mp)

# Initialize the cloud
cloud, converter = init_cloud(p, ethep)
print('Cloud initialized')

# Initialize the sph system
sph = init_sph(cloud, converter, p)
print('Sph initialized')

# Evolution of the system
print('System evolution...')
sph = evolve(sph, cloud, p)

# Save sph
parts = sph.gas_particles
write_set_to_file(parts, 'ressource/{}/{}.hdf'.format(p['name'], p['name']), overwrite_file=True)
print("SPH saved")

if p['vtk']:
    # Get density cube and velocity cubes
    view = set_view([0, 0, 0] | units.pc, p['radius'].value_in(units.pc))
    density, velocity, cells_lenght = sph_to_grid(sph, view, p['N'], p)
    print("SPH to grid conversion done") 

    # Save cloud
    savevtk(density, p['N'], '{}/{}full'.format(p['name'], p['name']))
    #savecloud_soc(density, p['N'], '{}/{}full'.format(p['name'], p['name']))

    print('cloud saved')

# Extract cores
find_cores(sph, p)

# Stop sph simulation
sph.stop()