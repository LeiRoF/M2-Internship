from functions import *
import os
from amuse.units import constants

#Constants
kb = constants.kB
mp = constants.proton_mass
G = constants.G

# Load parameters
p = simuini()

# Checking for necessary folders
if os.path.exists('output_files') == False:
    os.mkdir('output_files')
if os.path.exists('output_plots') == False:
    os.mkdir('output_plots')
if os.path.exists('ressource') == False:
    print('No ressource folder found. Exiting program')
    exit()
if os.path.exists('source') == False:
    print('No source folder found. Exiting program')
    exit()
if os.path.exists('ressource/{}'.format(p['name'])) == False:
    os.mkdir('ressource/{}'.format(p['name']))
if os.path.exists('output_files/{}'.format(p['name'])) == False:
    os.mkdir('output_files/{}'.format(p['name']))
if os.path.exists('output_plots/{}'.format(p['name'])) == False:
    os.mkdir('output_plots/{}'.format(p['name']))

# Init SPH
sph = init_sph_from_file(p)
print('SPH loaded')

if p['core'] == 2:
    # Extract cores
    find_cores(sph, p)

if p['core'] == 1:
    # Extract cores
    core_from_posfile(sph, p)