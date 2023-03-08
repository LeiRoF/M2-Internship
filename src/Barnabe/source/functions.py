import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pyvtk as vtk
import astrodendro
from astropy.io import fits
import os
import re

from amuse.datamodel import Particles, new_regular_grid
from amuse.ext.molecular_cloud import molecular_cloud
from amuse.community.fi.interface import Fi
from amuse.units import constants, nbody_system, units
from amuse.io import write_set_to_file, read_set_from_file

def sph_to_grid(sph_code, view, grid_size, p):
    """
    Thanks to Timothé Roland for the basis of this function

    Interpolate SPH representation on a defined cartesian grid

    Arguments:
    ----------
        sph_code: amuse.community.fi.interface.Fi
            sph code to make the interpolation of the sph representation

        view: amuse.units.quantities.ScalarQuantity list
            the (physical) region to sample [xmin, xmax, ymin, ymax, zmin, zmax] | units

        grid_size: int
            discretisation of the cube in pixels. 
        
        p: Dictionnary 
            Dictionnary containing the value of each parameter

    Return: 
    -------
        density map (array of shape grid_size)

        velocity maps (3 arrays of shape grid_size)
        
        length of cells in each direction (array in amuse units)
    """

    shape = (grid_size, grid_size, grid_size)
    size1D = grid_size **3
    axis_lengths = [0.0, 0.0, 0.0] | units.pc
    axis_lengths[0] = view[1] - view[0]
    axis_lengths[1] = view[3] - view[2]
    axis_lengths[2] = view[5] - view[4]
    grid = new_regular_grid(shape, axis_lengths)
    grid.x += view[0]
    grid.y += view[2]
    grid.z += view[4]
    speed = grid.z.reshape(size1D) * (0 | 1 / units.s) #convert into speed units

    rho, rhovx, rhovy, rhovz, rhoe = \
        sph_code.get_hydro_state_at_point(grid.x.reshape(size1D),
                                          grid.y.reshape(size1D),
                                          grid.z.reshape(size1D),
                                          speed, speed, speed)

    vx = rhovx / rho
    vy = rhovy / rho
    vz = rhovz / rho
    rho = rho.reshape(shape)
    vx = vx.reshape(shape)
    vy = vy.reshape(shape)
    vz = vz.reshape(shape)

    # Conversion of the quantities
    rho = rho.value_in(units.cm**-3 * units.kg)
    rho /= (p['mu']*1.6726219e-27)
    vx = vx.value_in(units.km * units.s**-1)
    vy = vy.value_in(units.km * units.s**-1)
    vz = vz.value_in(units.km * units.s**-1)

    # Construct velocity array
    vel = np.array((vx, vy, vz), dtype = np.float32)

    # physical size of each cells
    cells_length = [axis_lengths[i] / grid_size for i in range(3)]

    return rho.astype(np.float32), vel.astype(np.float32), cells_length


def init_cloud(p, ethep):
    """
    Function that create a cloud with uniform density and turbulences following
    Bate et al. (2003)

    Arguements:
    -----------
        p: Dictionnary 
            Dictionnary containing the value of each parameter

        ethep: float
            Thermal to potential energy ratio

    Return:
        cloud: amuse.datamodel.Particles
            initial structure of the cloud

        converter: amuse.units.nbody_system.nbody_to_si
            converter to convert nbody units to si units
    """

    # Initialise a gas cloud
    converter = nbody_system.nbody_to_si(p['mass'], p['radius'])
    cloud = molecular_cloud(targetN=p['npart'],
                            convert_nbody=converter,
                            seed=p['seed'],
                            ethep_ratio=ethep,
                            ekep_ratio=p['ekep'],
                            power=-3,
                            ).result
    
    return cloud, converter

def init_sph(cloud, converter, p):
    """
    Function that initialise the sph code

    Arguements:
    -----------
        cloud: amuse.datamodel.Particles
            initial structure of the cloud
        
        converter: amuse.units.nbody_system.nbody_to_si
            converter to convert nbody units to si units

        p: Dictionnary 
            Dictionnary containing the value of each parameter

    Return:
        sph: amuse.community.fi.interface.Fi
            The initialized sph code
    """

    # Initialise SPH code
    sph = Fi(converter, mode='openmp')

    # For isothermal cloud
    sph.parameters.gamma = 1.0  # default value: 1.6666667
    sph.parameters.isothermal_flag = True
    sph.parameters.integrate_entropy_flag = False

    # Other parameters
    sph.parameters.epsilon_squared = (0.05*p['radius'])**2
    sph.parameters.eps_is_h_flag = True
    sph.parameters.verbosity = 0
    sph.parameters.stopping_condition_maximum_density = p['maxdens'] * (2 * 1.6726219e-27 | units.kg)

    sph.gas_particles.add_particles(cloud)

    return sph

def init_sph_from_file(p):
    """
    Function that initialise the sph code from a file

    Arguements:
    -----------
        p: Dictionnary 
            Dictionnary containing the value of each parameter

    Return:
        sph: amuse.community.fi.interface.Fi
            The initialized sph code
    """

    converter = nbody_system.nbody_to_si(p['mass'], p['radius'])

    # Initialise SPH code
    sph = Fi(converter, mode='openmp')

    # For isothermal cloud
    sph.parameters.gamma = 1.0  # default value: 1.6666667
    sph.parameters.isothermal_flag = True
    sph.parameters.integrate_entropy_flag = False

    sph.parameters.epsilon_squared = (0.05*p['radius'])**2
    sph.parameters.eps_is_h_flag = True
    sph.parameters.verbosity = 0

    try:
        parts = read_set_from_file('ressource/{}/{}.hdf'.format(p['name'], p['name']), 'amuse', close_file=True)
    except:
        print('No saved sph simulation found. Please verify the parameter "name"')
        exit()

    sph.gas_particles.add_particles(parts)

    return sph

def evolve(sph, cloud, p):
    """
    Function that evolve the cloud until a given time defined by a fraction
    of the free-fall time or while the maximum density is under a given value

    Arguements:
        sph: amuse.community.fi.interface.Fi
            The sph code

        cloud: amuse.datamodel.Particles
            initial structure of the cloud
        
        p: Dictionnary 
            Dictionnary containing the value of each parameter  

    Return:
        sph: amuse.community.fi.interface.Fi
            The sph code in its current state
    """
    # Constant
    G = constants.G

    # System time step
    fft = ((np.pi**2 * p['radius']**3)/(8*G*p['mass']))**0.5   # freefall time
    sph.parameters.timestep = fft/1000

    # Set communication
    to_cloud = sph.gas_particles.new_channel_to(cloud, ['density', 'h_smooth',
                                                        'mass', 'u',
                                                        'x', 'y', 'z',
                                                        'vx', 'vy', 'vz', 'pressure'])

    # Evolution
    dt = p['tfact']*fft / 100
    time = dt
    progress = 0
    while (time < p['tfact']*fft):
        sph.evolve_model(time)
        time += dt
        progress += 1
        print("progress: {}%".format(progress), end = "\r") 

        density_limit_detection = sph.stopping_conditions.density_limit_detection
        density_limit_detection.enable()

        if density_limit_detection.is_set():
            print("\n") 
            print('Maximum density reached! Evolution time: {} Myr'.format(time.value_in(units.Myr)))
            return sph

    print("\n") 
    print('Maximum simulation time reached! Evolution time: {} Myr'.format(time.value_in(units.Myr)))
    return sph


###############################################################################
#                   A SUPPRIMER A LA FIN
###############################################################################
def evolve_anim(sph, p, frame):
    """
    Function that evovle the system and that save particle position
    at regular time interval

    Arguements:
        sph: amuse.community.fi.interface.Fi
            The sph code
        
        

        frame: int
            Number of snapshot of the system

    Return:
        sph: amuse.community.fi.interface.Fi
            The sph code in its current state
    """

    # Constant
    G = constants.G

    # System time step
    fft = ((np.pi**2 * p['radius']**3)/(8*G*p['mass']))**0.5   # freefall time
    sph.parameters.timestep = fft/1000

    # Evolution
    dt = p['tfact']*fft / frame
    time = 0 | units.s
    progress = 0
    ind = 0
    truemax = 0

    # Cube limits
    view = set_view([0, 0, 0] | units.pc, p['radius'].value_in(units.pc))

    mdens_list = np.zeros((frame, p['N'], p['N']))

    while time < p['tfact']*fft:
        sph.evolve_model(time)
        print("Time: {} Myr".format(time.value_in(units.Myr)))
        # Get density cube and velocity cubes

        # Save sph
        parts = sph.gas_particles
        write_set_to_file(parts, 'ressource/{}/{}_{}.hdf'.format(p['name'], p['name'], ind), overwrite_file=True)
        print("SPH saved")

        # density, _, _ = sph_to_grid(sph, view, p['N'], p)

        # column_dens(density, p['N'], '{}/{}_{}'.format(p['name'], p['name'], ind), p['radius'])

        time += dt
        progress += 100/frame
        ind += 1

        print("progress: {}%".format(progress), end = "\r") 
    print("\n") 
    print('Evolution time: {} Myr'.format(time.value_in(units.Myr)))

    return sph

def savesph(sph, name):
    """
    Function that save the position, velocities and density of the particles in a binary file

    Arguments:
    ----------
        sph: amuse.community.fi.interface.Fi
            The sph code

        name: str
            prefix of the file
    """

    fp = open('ressource/{}.sph'.format(name), 'wb')
    posx = sph.gas_particles.x.value_in(units.pc)
    posy = sph.gas_particles.y.value_in(units.pc)
    posz = sph.gas_particles.z.value_in(units.pc)
    density = sph.gas_particles.density.value_in(units.kg * units.m**-3)
    vx = sph.gas_particles.vx.value_in(units.m * units.s**-1)
    vy = sph.gas_particles.vy.value_in(units.m * units.s**-1)
    vz = sph.gas_particles.vz.value_in(units.m * units.s**-1)

    np.array([posx, posy, posz, vx, vy, vz, density], np.float32).tofile(fp)
    fp.close()

def loadsph(name, nbpart):
    """
    Function that load a file contains the position of the particle of an SPH
    simulation

    Arguments:
    ----------
        name: str
            prefix of the file

        nbparts: int
            number of particles used in the SPH simulation

    Return:
    -------
        pos: float32 numpy.array
            array containing the position of each particles
    """

    fp = open('ressource/{}.sph'.format(name), 'r')
    sph = np.fromfile(fp, np.float32).reshape(7, nbpart)
    return sph

###############################################################################
###############################################################################

def savecloud_soc(array, N, name):
    """
    Function that save the density structure of hte cloud such as it can be read
    by SOC

    Arguements:
    -----------
        array: numpy.array
            density cube
        
         N: int
            The resolution of the output cube(s)
        
         name: str
            prefix of the file       
    """

    # CLoud file initialization
    fp  =  open('ressource/{}.cloud'.format(name), 'w')
    np.asarray([N, N, N, 1, N*N*N], np.int32).tofile(fp)  #  NX, NY, NZ, LEVELS, CELLS
    np.asarray([N*N*N,], np.int32).tofile(fp)             #  cells on the first (and only) level
    array = np.swapaxes(array, 0, 2)
    array.tofile(fp)
    fp.close() 

def savevel(array, name):
    """
    Function that save the velocities within the cloud

    Arguements:
    -----------
        array: numpy.array
            velocity cube

        name: str
            prefix of the file

    """

    # Vel file initialization
    fp  =  open('ressource/{}.vel'.format(name), 'w')
    array.tofile(fp)
    fp.close() 

def loadcloud(name):
    """
    Function that load a density cube

    Arguement:
    ----------
        name: str
            prefix of the file

    Return:
    -------
        cloud: numpy.array
            density cube
        (NZ, NY, NX): int tupple
            dimension of each axis of the cube
    """
    # Read SOC cloud
    fp  =  open('ressource/{}.cloud'.format(name), 'r')
    NZ, NY, NX, LEVELS, CELLS, _ = np.fromfile(fp, np.int32, 6)
    cloud = np.fromfile(fp, np.float32).reshape(NZ, NY, NX)

    return cloud, (NZ, NY, NX)

def loadvel(N, name):
    """
    Function that load a velocity cube

    Arguments:
    ----------
         N: int
            The resolution of the output cube(s)

        name: str
            prefix of the file

    Return:
        vel: numpy.array
            velocity cube       
    """
    # Read velocity file
    fp  =  open('ressource/{}.vel'.format(name), 'r')
    vel = np.fromfile(fp, np.float32).reshape(3, *N)

    return vel

def loadfits(name):
    """
    Function that loads a FITS file

    Argument:
    ---------
        name: str
            Name of the FITS file

    Return:
    -------
        out: numpy array
            Data contained in the FITS file

    """
    fitmap = fits.open('output_files/{}.fits'.format(name))
    out = fitmap[0].data.astype(np.float64)
    fitmap.close()

    return out

def savefits(array, name):
    """
    Function that save data in a FITS file

    Arguments:
    ----------
        array: numpy array
            Data to save

        name: name of the FITS file
    """

    hdu = fits.PrimaryHDU(array)
    hdu.writeto('output_files/{}.fits'.format(name), overwrite=True)

def loadtemp(name):
    """
    Function that load the temperature file returned by SOC

    Arguement:
    ----------
        name: str
            prefix of the file    

    Return:
    -------
        T: numpy.array
            Temperature cube
    """

    # Load true temperature file
    fp       = open('output_files/{}.T'.format(name), 'rb')            # open the file with dust temperatures
    NX, NY, NZ, _, _, _ = np.fromfile(fp, np.int32, 6)
    T        = np.fromfile(fp, np.float32).reshape(NZ, NY, NX) # T is simply a 64x64x64 cell cube    

    return T

def make_cloud(cloud, velocities, p):
    """
    Functions that return an array containing the characteristics of the cloud
    arranged such as LOC can read it

    Arguments:
    ----------
        cloud: amuse.units.quantities.ScalarQuantity array
            Density cube

        velocities: amuse.units.quantities.ScalarQuantity array
            Velocity cube

        Temp: float
            Kinetic temperature

         N: int
            The resolution of the output cube(s)       

    Return:
        C: numpy array
            array containing the different characteristics of the cloud
    """

    # Initialize cloud file
    C = np.ones((p['N'], p['N'], p['N'], 7), np.float32)

    # Set the values in the cube
    C[:, :, :, 0] = cloud
    C[:,:,:,1]  = p['T'].value_in(units.K)
    C[:,:,:,2]  *= 1.0                   # leave microturbulence at 1 km/s
    C[:,:,:,3]   = np.swapaxes(velocities[0], 0, 2)
    C[:,:,:,4]   = np.swapaxes(velocities[1], 0, 2)
    C[:,:,:,5]   = np.swapaxes(velocities[2], 0, 2)
    C[:,:,:,6]   = cloud**2.45 / (3e8 + cloud**2.45) # abundance values will be scaled later


    return C

def savecloud_loc(array, N, name):
    """
    Function that save a file containing the cloud and formatted such as LOC can
    read it

    Arguments:
    ----------
        array: numpy.array
            array containing the different characteristics of the cloud

        N: int
            The resolution of the output cube(s)

        name: str
            prefix of the file
    """

    # Write the actual file
    fp = open('ressource/{}_loc.cloud'.format(name), 'wb')
    np.asarray([*N], np.int32).tofile(fp)
    array.tofile(fp)
    fp.close()

def column_dens(cloud, N, name, csize, out=False):
    """
    Function that compute the column density map of the cloud from a density cube

    Arguments:
    ----------
        cloud: amuse.units.quantities.ScalarQuantity array
            Density cube

        N: int
            The resolution of the input cube

        name: str
            prefix of the file
        
        csize: amuse.units.quantities.ScalarQuantity
            Size of the data cube

    Return (if out = True):
    -----------------------
        fig: matplotlib.figure
            Reference to the column density map figure

        ax: matplotlib.axes
            The axes of the column density map figure
    """

    dl = 2*csize.value_in(units.cm)/N
    c_dens = np.sum(cloud, axis=2) * dl

    print(np.max(c_dens))

    R = csize.value_in(units.pc)
    fig, ax = plt.subplots(figsize=(7, 5))

    if out:
        im = ax.imshow(c_dens, extent=[-R, R, -R, R], origin='lower', cmap='CMRmap')
        fig.colorbar(im, label=r'Column density ($H_2.cm^{-2}$)')
        ax.set_xlabel(r'Distance (pc)')
        ax.set_ylabel(r'Distance (pc)')
        return fig, ax
    else:
        im = ax.imshow(c_dens, extent=[-R, R, -R, R], origin='lower', cmap='CMRmap', vmin= 0.0, vmax=2e23)
        #fig.colorbar(im, label=r'Column density ($H_2.cm^{-2}$)')
        #ax.set_xlabel(r'Distance (pc)')
        #ax.set_ylabel(r'Distance (pc)')
        plt.axis('off')
        fig.savefig('output_plots/{}.png'.format(name), bbox_inches='tight')  #_column_density_true.pdf
        plt.clf()
        savefits(c_dens, '{}_cd_true'.format(name))

def max_dens(cloud, N, name, csize, out=False):
    """
    Function that compute the max density map of the cloud from a density cube

    Arguments:
    ----------
        cloud: amuse.units.quantities.ScalarQuantity array
            Density cube

        N: int
            The resolution of the input cube

        name: str
            prefix of the file
        
        csize: amuse.units.quantities.ScalarQuantity
            Size of the data cube

    Return (if out = True):
    -----------------------
        fig: matplotlib.figure
            Reference to the maximum density map figure

        ax: matplotlib.axes
            The axes of the maximum density map figure
    """

    m_dens = np.max(cloud, axis=2)

    R = csize.value_in(units.pc)
    fig, ax = plt.subplots(figsize=(7, 5))

    if out:
        im = ax.imshow(m_dens, extent=[-R, R, -R, R], origin='lower', cmap='CMRmap')
        fig.colorbar(im, label=r'Maximum density ($H_2.m^{-3}$)')
        ax.set_xlabel(r'Distance (pc)')
        ax.set_ylabel(r'Distance (pc)')
        return fig, ax
    else:
        im = ax.imshow(m_dens, extent=[-R, R, -R, R], origin='lower', cmap='CMRmap')
        fig.colorbar(im, label=r'Maximum density ($H_2.cm^{-3}$)')
        ax.set_xlabel(r'Distance (pc)')
        ax.set_ylabel(r'Distance (pc)')
        fig.savefig('output_plots/{}_max_density.pdf'.format(name), bbox_inches='tight')
        plt.clf()
        savefits(m_dens, '{}_md_true'.format(name))

def savevtk(cloud, N, name):
    """
    Function that save the density cube in a .vtk file

    Arguments:
    ----------
        cloud: amuse.units.quantities.ScalarQuantity array
            Density cube

        N: int
            The resolution of the output cube(s)

        name: str
            prefix of the file
    """

    pointdata = vtk.PointData((vtk.Scalars(cloud.reshape(N**3,), name='Scalars0')))
    data = vtk.VtkData(vtk.StructuredPoints([N, N, N]), pointdata)
    data.tofile('ressource/{}'.format(name), 'binary')

def set_view(pos, offset):
    """
    Set the view, i.e the boundaries of the cube(s)

    Arguments:
    ----------
        pos: amuse.units.quantities.ScalarQuantity list
            position [x, y, z] of the center of the cube(s)

        offset: float
            1/2 size of the cube(s) 

    Return:
    -------
        amuse.units.quantities.ScalarQuantity list
            List containing the boundaries of the cube(s)

    """
    # parsec to meter converter
    pc2m = (1.0 | units.pc).value_in(units.m)

    # Half size of the cube
    offset = offset * pc2m | units.m

    # Boundaries
    xview = [pos[0] - offset, pos[0] + offset]
    yview = [pos[1] - offset, pos[1] + offset]
    zview = [pos[2] - offset, pos[2] + offset]

    return [*xview, *yview, *zview]

def simuini():
    """
    This function read the simulation parameters from the configuration file

    Return:
    -------
        out: dictonnary
            Dictionnary containing the value of each parameter
    """

    out = {}
    for line in open('parameters.ini').readlines():
        s = line.split('#')[0].split()
        key = s[0]
        del s[0]
        a = s

        if (key.find('name')==0):
            out['name'] = str(a[0])

        if (key.find('nbcore')==0):
            out['nbcore'] = int(a[0])

        if (key.find('mass')==0):
            out['mass'] = float(a[0]) | units.MSun

        if (key.find('radius')==0):
            out['radius'] = float(a[0]) | units.pc

        if (key.find('N')==0):
            out['N'] = int(a[0])

        if (key.find('npart')==0):
            out['npart'] = int(a[0])

        if (key.find('tfact')==0):
            out['tfact'] = float(a[0])  

        if (key.find('mu')==0):
            out['mu'] = float(a[0])

        if (key.find('T')==0):
            out['T'] = float(a[0]) | units.K

        if (key.find('ekep')==0):
            out['ekep'] = float(a[0])

        if (key.find('dendro')==0):
            out['dendro'] = int(a[0])

        if (key.find('sleaf')==0):
            out['sleaf'] = float(a[0]) | units.pc

        if (key.find('seed')==0):
            if a[0] == 'None':
                out['seed'] = None
            else:
                out['seed'] = int(a[0])
        
        if (key.find('nbgpu')==0):
            out['nbgpu'] = int(a[0])
        
        if (key.find('nbcpu')==0):
            out['nbcpu'] = int(a[0])

        if (key.find('nbthread')==0):
            out['nbthread'] = int(a[0])
        
        if (key.find('uncertmc')==0):
            out['uncertmc'] = int(a[0])

        if (key.find('maxdens')==0):
            out['maxdens'] = float(a[0]) | units.cm**-3

        if (key.find('nbsocket')==0):
            out['nbsocket'] = int(a[0])

        if (key.find('soc')==0):
            out['soc'] = int(a[0])

        if (key.find('loc')==0):
            out['loc'] = int(a[0]) 

        if (key.find('vleaf')==0):
            out['vleaf'] = float(a[0])  

        if (key.find('core')==0):
            out['core'] = int(a[0])   

        if (key.find('molecule')==0):
            out['molecule'] = a  

        if (key.find('resloc')==0):
            out['resloc'] = int(a[0])

        if (key.find('abundance')==0):
            out['abundance'] = a

        if (key.find('transition')==0):
            out['transition'] = a
        
        if (key.find('vtk')==0):
            out['vtk'] = int(a[0])
        
        if (key.find('distance')==0):
            out['distance'] = float(a[0])

    return out

def SOCini(csize, N, name, gpu=0):
    """
    This function write an file containing the parameters for SOC 

    Arguments:
    ----------
        csize: amuse.units.quantities.ScalarQuantity
            Size of the data cube
        
        N: int
            The resolution of the output cube(s)

        name: str
            Prefix of the name output file

        gpu: int
            Index of the GPU to use
    """

    inifile = [
    "gridlength      {}".format(2*csize.value_in(units.pc)/N)   , # root grid cells size
    "cloud           ressource/{}.cloud".format(name)           , # density field (reference to a file)
    "mapping         {} {} 1.0".format(N, N)                    , # output NxN pixels, pixel size == root-grid cell size
    "density         1.0"                                       , # scale values read from tmp.cloud
    "seed           -1.0"                                       , # random seed for random numbers
    "directions      0.0 -90.0"                                 , # observer in direction (theta, phi)
    "optical         ressource/aSilx.dust"                      , # dust optical parameters
    "dsc             ressource/aSilx.dsc  2500"                 , # dust scattering function
    "bgpackets       1000000"                                   , # photon packages simulated from the background
    "background      ressource/bg_intensity.bin"                , # background intensity at the
    "iterations      1"                                         , # one iteration is enough
    "prefix          {}".format(name)                           , # prefix for output files
    "absorbed        ressource/absorbed.data"                   , # save absorptions to a file
    "emitted         ressource/{}_emitted.data".format(name)    , # save dust emission to a file
    "noabsorbed"                                                , # actually, we integrate absorptions on the fly and skip the *file*
    "temperature     output_files/{}.T".format(name)            , # save dust temperatures
    "device          g"                                         , # run calculations on a GPU
    "platform        0 {}".format(gpu)                          , # Choose on which gpu to run
    "CLT"                                                       , # temperature caclulations done on the device
    "CLE"                                                       , # emission calculations done on the device
    "fits 0.0 0.0 output_files/{}".format(name)                 , # Save fits whith name "name"
    "mapum 100.0 160.0 250.0 350.0 500.0"                       , # Save wavelenghts
    ]

    np.savetxt('ressource/{}_soc.ini'.format(name), inifile, fmt="%s")

def LOCini(p, core, size, gpu=0, mol=0):
    """
    This function write an file containing the parameters for SOC 

    Arguments:
    ----------
        p: dictonnary
            Dictionnary containing the value of each parameter

        core: int
            Number of the core

        gpu: int
            Index of the GPU to use
    """

    angle = (3600 * 180 / np.pi) * size/p['distance']

    transitions = re.split('[- ,]', p['transition'][mol])

    name = '{}/{}core{}/{}core{}'.format(p['name'], p['name'], core, p['name'], core)

    inifile = [
    "cloud          ressource/{}_loc.cloud".format(name)             , # cloud defined on a Cartesian grid
    "distance       {}".format(p['distance'])                        , # cloud distance [pc]
    "angle          {}".format(angle / p['N'])                       , # model cell size  [arcsec]
    "molecule       ressource/{}.dat".format(p['molecule'][mol])     , # name of the Lamda file
    "density        1.0"                                             , # scaling of densities in the cloud file
    "temperature    1.0"                                             , # scaling of Tkin values in the cloud file
    "fraction       {}".format(p['abundance'][mol])                  , # scaling o the fraction abundances
    "velocity       1.0"                                             , # scaling of the velocity
    "sigma          0.1"                                             , # scaling o the microturbulence
    "isotropic      2.73"                                            , # Tbg for the background radiation field
    "levels         10"                                              , # number of energy levels in the calculations
    "uppermost      3"                                               , # uppermost level tracked for convergence
    "iterations     5"                                               , # number of iterations
    "nside          2"                                               , # NSIDE parameter determining the number of rays (angles)
    "direction      0.0 -90.0"                                       , # theta, phi [deg] for the direction towards the observer
    "points         {} {}".format(p['resloc'], p['resloc'])          , # number of pixels in the output maps
    "grid           {}".format(angle / p['resloc'])                  , # map pixel size [arcsec]
    "spectra        {}".format(' '.join(transitions))                , # spectra written for transitions == upper lower (pair of integers)
    "transitions    {}".format(' '.join(transitions))                , # Tex saved for transitions (upper and lower levels of each transition)
    "bandwidth      6.0"                                             , # bandwidth in the calculations [km/s]
    "channels       128"                                             , # number of velocity channels
    "prefix         output_files/{}".format(name)                    , # prefix for output files
    "load           ressource/{}.save".format(name)                  , # load level populations
    "save           ressource/{}.save".format(name)                  , # save level populations
    "stop          -1.0"                                             , # stop if level-populations change below a threshold
    "gpu            1"                                               , # gpu>0 => use the first available gpu
    "tausave        1"                                               , # Save optical depth
    "platform       0 {}".format(gpu)                                , # Choose on which gpu to run
    ]

    np.savetxt('ressource/{}_{}_loc.ini'.format(name, p['molecule'][mol]), inifile, fmt="%s")


def find_cores(sph, p):
    """
    This function exctract regions of high density using a dendrogram and save
    them as data cubes

    Arguments:
    ----------
        sph: amuse.community.fi.interface.Fi
            The sph code

        p: Dictionnary 
            Dictionnary containing the value of each parameter
    """
    import time
    # Middle of the cube
    N2 = p['dendro']/2 - 0.5

    # Boundary of the cube used to build the dendrogram
    view = [-p['radius'], p['radius'], -p['radius'], p['radius'], -p['radius'], p['radius']]

    # Get density cube and velocity cubes
    density, _, _ = sph_to_grid(sph, view, p['dendro'], p)
    print("SPH to grid conversion done") 

    # Size of 1 pixel
    pixs = 2*p['radius'].value_in(units.pc) / p['dendro']

    # Minimum number of pixels for a leaf
    npix = round(p['sleaf'].value_in(units.pc) / pixs)
    if npix == 0:
        print('Minimum number of pixel of a leaf is 0! It will be set automatically to 1')
        npix = 1
    else:
        print ('Minimum number of pixel of a leaf:', npix)

    # Make dendrogram
    custom_independent = astrodendro.pruning.min_peak(p['vleaf'] * np.max(density))
    start = time.time()
    d = astrodendro.Dendrogram.compute(density, min_npix=npix, is_independent=custom_independent)
    end = time.time()
    print("Time to build the dendrogram:".format(end - start))

    # Sort leaves
    leaves = d.trunk[0].sorted_leaves(reverse = True, subtree=True)
    leaves = sorted(leaves, key=lambda leave: leave.get_peak()[1], reverse=True)

    for i in range(len(leaves)):
        print(leaves[i].get_peak())

    if len(leaves) < p['nbcore']:
        print('Desired number of core is two high.')
        print('It has been automatically set to {}'.format(len(leaves)))
        p['nbcore'] = len(leaves)

    # Compute cloud column density map
    fig, ax = column_dens(density, p['dendro'], p['name'] + 'core_pos', p['radius'], out=True)
    # Compute cloud maximum density map
    fig1, ax1 = max_dens(density, p['dendro'], p['name'] + 'core_pos', p['radius'], out=True)

    gpu = 0

    adp = np.zeros((p['nbcore'], 4))

    # Loop that extract the cores
    for n in range(p['nbcore']):
        print('core {}:'.format(n+1))
        name = '{}core{}'.format(p['name'], n+1)
        folder = '{}/{}'.format(p['name'], name)
        
        if os.path.exists('ressource/{}'.format(folder)) == False:
            os.mkdir('ressource/{}'.format(folder))
        if os.path.exists('output_files/{}'.format(folder)) == False:
            os.mkdir('output_files/{}'.format(folder))
        if os.path.exists('output_plots/{}'.format(folder)) == False:
            os.mkdir('output_plots/{}'.format(folder))

        # Get nth leaf
        values = leaves[n].values()
        indices = np.array(leaves[n].indices())
        indices = indices.reshape(3, len(indices[0]))

        # Get minimum and maximum pixel position
        min_pix = np.array([min(indices[0, :]), min(indices[1, :]), min(indices[2, :])])
        max_pix = np.array([max(indices[0, :]), max(indices[1, :]), max(indices[2, :])])

        # Pixel to distance (pc)
        min_dists = min_pix
        max_dists = max_pix

        # Cube size
        dist_max = np.max(max_dists - min_dists) * pixs

        # Avoid cube size of zero when leaf size equals 1 pixel
        if dist_max == 0:
            dist_max = (npix * pixs)

        dm = dist_max/2.0 | units.pc

        # Cube center
        pos = np.array([np.mean(indices[0, :]), np.mean(indices[1, :]), np.mean(indices[2, :])])
        center_pos = (pos - N2) * pixs | units.pc

        # Center correction
        parts = sph.gas_particles
        mask = (parts.x >= center_pos[0]-dm) & (parts.x <= center_pos[0]+dm) \
         & (parts.y >= center_pos[1]-dm) & (parts.y <= center_pos[1]+dm) \
         & (parts.z >= center_pos[2]-dm) & (parts.z <= center_pos[2]+dm)
        
        parts = parts[mask]
        booldens = parts.density.value_in(units.g * units.cm**-3) > 0.25*np.max(parts.density.value_in(units.g * units.cm**-3))
        correct = 5
        while len(parts[booldens]) > 0.5*len(parts) and correct > 1:
            correct -= 1
            dist_max *= 2.0
            dm *= 2.0
            # Center correction
            parts = sph.gas_particles
            mask = (parts.x >= center_pos[0]-dm) & (parts.x <= center_pos[0]+dm) \
            & (parts.y >= center_pos[1]-dm) & (parts.y <= center_pos[1]+dm) \
            & (parts.z >= center_pos[2]-dm) & (parts.z <= center_pos[2]+dm)
            
            parts = parts[mask]
            booldens = parts.density.value_in(units.g * units.cm**-3) > 0.25*np.max(parts.density.value_in(units.g * units.cm**-3))

        index = np.argmax(parts.density)

        center_pos[0] = parts[index].x
        center_pos[1] = parts[index].y
        center_pos[2] = parts[index].z

        print('    pos pix:', pos)
        print('    dist:', dist_max)
        print('    pos cube:', center_pos)

        adp[n] = [*center_pos.value_in(units.pc), dist_max/2.0]

        # Set the boundary of the data cube(s)
        view = set_view(center_pos, dist_max/2.0)

        # Get density cube and velocity cubes
        density, velocity, cells_lenght = sph_to_grid(sph, view, p['N'], p)
        print("    SPH to grid conversion done")

        # Compute column density map of the core
        column_dens(density, p['N'], '{}/{}'.format(folder, name), dist_max/2.0 | units.pc)
        # Compute maximum density map of the core
        max_dens(density, p['N'], '{}/{}'.format(folder, name), dist_max/2.0 | units.pc)
        print("    Maps computed")

        # Save core
        savecloud_soc(density, p['N'], '{}/{}'.format(folder, name))
        savevel(velocity, '{}/{}'.format(folder, name))

        if p['vtk']:
            savevtk(density, p['N'], '{}/{}'.format(folder, name))

        if gpu > p['nbgpu'] - 1:
            gpu = 0

        SOCini(dist_max/2.0 | units.pc, p['N'], '{}/{}'.format(folder, name), gpu)

        # Add position of the core on the cloud column density map
        pos1 = center_pos.value_in(units.pc)

        rect = Rectangle((pos1[1]-dist_max/2.0,
                          pos1[0]-dist_max/2.0),
                          dist_max,
                          dist_max,
                          linewidth=1,
                          edgecolor='r',
                          facecolor='none')
        ax.add_patch(rect)
        ax.text(pos1[1], pos1[0], '{}'.format(n+1), color='w')

        rect = Rectangle((pos1[1]-dist_max/2.0,
                          pos1[0]-dist_max/2.0),
                          dist_max,
                          dist_max,
                          linewidth=1,
                          edgecolor='r',
                          facecolor='none')
        ax1.add_patch(rect)
        ax1.text(pos1[1], pos1[0], '{}'.format(n+1), color='w')

        print('    Saved')
        gpu += 1
    
    fig.savefig('output_plots/{}/{}corepos_cd.pdf'.format(p['name'], p['name']), bbox_inches='tight')
    fig1.savefig('output_plots/{}/{}corepos_md.pdf'.format(p['name'], p['name']), bbox_inches='tight')

    np.savetxt('ressource/{}/{}_corepos.dat'.format(p['name'], p['name']), adp)


def core_from_posfile(sph, p):
    """
    This function exctract regions of the cloud using a file containing the
    position and size of the each regions (in parsec) and save
    them as data cubes

    Arguments:
    ----------
        sph: amuse.community.fi.interface.Fi
            The sph code

        p: Dictionnary 
            Dictionnary containing the value of each parameter
    """

    # Load cubes position in the cloud
    pos_dist_array = np.loadtxt('ressource/{}/{}_corepos.dat'.format(p['name'], p['name'])) | units.pc

    # Boundary of the cube used to build the dendrogram
    view = [-p['radius'], p['radius'], -p['radius'], p['radius'], -p['radius'], p['radius']]

    # Get density cube and velocity cubes
    density, _, _ = sph_to_grid(sph, view, p['dendro'], p)
    print("SPH to grid conversion done") 

    if len(pos_dist_array) < p['nbcore']:
        print('Desired number of core is too high.')
        print('It has been automatically set to {}'.format(len(pos_dist_array)))
        p['nbcore'] = len(pos_dist_array)

    # Compute cloud column density map
    fig, ax = column_dens(density, p['dendro'], p['name'] + 'core_pos', p['radius'], out=True)
    # Compute cloud maximum density map
    fig1, ax1 = max_dens(density, p['dendro'], p['name'] + 'core_pos', p['radius'], out=True)

    gpu = 0

    # Loop that extract the cores
    for n in range(p['nbcore']):
        print('core {}:'.format(n+1))
        name = '{}core{}'.format(p['name'], n+1)
        folder = '{}/{}'.format(p['name'], name)
        
        if os.path.exists('ressource/{}'.format(folder)) == False:
            os.mkdir('ressource/{}'.format(folder))
        if os.path.exists('output_files/{}'.format(folder)) == False:
            os.mkdir('output_files/{}'.format(folder))
        if os.path.exists('output_plots/{}'.format(folder)) == False:
            os.mkdir('output_plots/{}'.format(folder))

        # Cube size
        dist_max = pos_dist_array[n, 3].value_in(units.pc)

        center_pos = pos_dist_array[n, 0:3]

        print('    dist:', dist_max*2.0)
        print('    pos cube:', center_pos)

        
        # Set the boundary of the data cube(s)
        view = set_view(center_pos, dist_max)

        # Get density cube and velocity cubes
        density, velocity, cells_lenght = sph_to_grid(sph, view, p['N'], p)
        print("    SPH to grid conversion done")

        # Compute column density map of the core
        column_dens(density, p['N'], '{}/{}'.format(folder, name), dist_max | units.pc)
        # Compute maximum density map of the core
        max_dens(density, p['N'], '{}/{}'.format(folder, name), dist_max | units.pc)
        print("    Maps computed")

        # Save core
        savecloud_soc(density, p['N'], '{}/{}'.format(folder, name))
        savevel(velocity, '{}/{}'.format(folder, name))

        if p['vtk']:
            savevtk(density, p['N'], '{}/{}'.format(folder, name))

        if gpu > p['nbgpu'] - 1:
            gpu = 0

        SOCini(dist_max | units.pc, p['N'], '{}/{}'.format(folder, name), gpu)

        # Add position of the core on the cloud column density map
        pos1 = center_pos.value_in(units.pc)

        rect = Rectangle((pos1[1]-dist_max,
                          pos1[0]-dist_max),
                          2.0*dist_max,
                          2.0*dist_max,
                          linewidth=0.1,
                          edgecolor='r',
                          facecolor='none')
        ax.add_patch(rect)
        ax.text(pos1[1]+dist_max, pos1[0], '{}'.format(n+1), color='w', fontsize='5')

        rect = Rectangle((pos1[1]-dist_max,
                          pos1[0]-dist_max),
                          2.0*dist_max,
                          2.0*dist_max,
                          linewidth=0.1,
                          edgecolor='r',
                          facecolor='none')
        ax1.add_patch(rect)
        ax1.text(pos1[1]+dist_max, pos1[0], '{}'.format(n+1), color='w', fontsize='5')

        print('    Saved')
        gpu += 1
    
    fig.savefig('output_plots/{}/{}corepos_cd.pdf'.format(p['name'], p['name']), bbox_inches='tight')
    fig1.savefig('output_plots/{}/{}corepos_md.pdf'.format(p['name'], p['name']), bbox_inches='tight')
    

def load_spectra_3d(filename):
    """
    Function from Juvela Mika
    Read spectra written by LOC.py (LOC_OT.py; 3D models)
    Argument:
    ---------
        filename: str
            name of the spectrum file

    Return:
        V: numpy.ndarray
            vector of velocity values, one per channel

        S: nummy.ndarray
            spectra as a cube S[NRA, NDE, NCHN] for NRA times
            NDE points on the sky,
            NCHN is the number of velocity channels
    """

    fp              =  open(filename, 'rb')
    NRA, NDE, NCHN  =  np.fromfile(fp, np.int32, 3)
    V0, DV          =  np.fromfile(fp, np.float32, 2)
    SPE             =  np.fromfile(fp, np.float32).reshape(NRA, NDE,2+NCHN)
    OFF             =  SPE[:,:,0:2].copy()
    SPE             =  SPE[:,:,2:]
    fp.close()
    return V0+np.arange(NCHN)*DV, SPE

def save_spectra_3d(vel, spe, name):
    """
    Save PPV cubes containing spectra for differnet regions

    Arguments:
    ----------
        vel: numpy.ndarray
            vector of velocity values, one per channel
        
        spe: numpy.ndarray
            spectra as a PPV cube

        name: str
            name of the spectrum file
    """

    fp = open(name, 'w')

    NY, NX, NCHN = spe.shape

    np.asarray([NY, NX, NCHN], np.int32).tofile(fp)
    np.asarray(vel, np.float32).tofile(fp)
    spe.tofile(fp)
    fp.close()

def load_spectra_3d_uncert(name):
    """
    Read PPV cube containing spectra from different regions

    Argument:
    ---------
        name: str
            name of the spectrum file
    
    Return:
    -------
        vel: numpy.ndarray
            vector of velocity values, one per channel
            
        spe: numpy.ndarray
            spectra as a PPV cube
    """

    fp = open(name, 'rb')
    NY, NX, NCHN = np.fromfile(fp, np.int32, 3)
    vel = np.fromfile(fp, np.float32, NCHN)
    spe = np.fromfile(fp, np.float32).reshape(NY, NX, NCHN)
    return vel, spe
