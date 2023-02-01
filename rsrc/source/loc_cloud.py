from functions import *
import concurrent.futures
import sys

def loadini(name):
    """
    
    """

    for line in open('ressource/{}_soc.ini'.format(name)).readlines():
        s = line.split('#')[0].split()
        key, a  =  s[0], s[1]
        if (key.find('gridlength')==0):
            return float(a)

def loc_cloud(name, p):
    """
    Function that create a LOC compatible data cube

    Argument:
    ---------
        name: str
            Path/Name of the object for which a data cube will be created
    """

    # Read cloud file
    cloud, N = loadcloud(name)
    print('Cloud load')

    # Read velocities
    velocities = loadvel(N, name)
    print('Velocities load')

    # Make LOC readable cloud
    new_cloud = make_cloud(cloud, velocities, p)
    print('LOC cloud done')

    # Save new cloud
    savecloud_loc(new_cloud, N, name)
    print('LOC cloud saved')

if __name__ == '__main__':

    # Parameters
    p = simuini()

    with concurrent.futures.ProcessPoolExecutor(max_workers=p['nbthread']*p['nbsocket']) as executor:
        for core in range(1, p['nbcore']+1):
            executor.submit(loc_cloud, '{}/{}core{}/{}core{}'.format(p['name'], p['name'], core, p['name'], core), p)

    # Load position and half-size of the cubes
    pos_dist_array = np.loadtxt('ressource/{}/{}_corepos.dat'.format(p['name'], p['name']))
    
    gpu = 0
    for core in range(1, p['nbcore']+1):
        name = '{}core{}'.format(p['name'], core)
        folder = '{}/{}'.format(p['name'], name)        

        if gpu > p['nbgpu'] - 1:
            gpu = 0
        
        for mol in range(len(p['molecule'])):
            LOCini(p, core, pos_dist_array[core-1, 3]*2.0, gpu, mol)

        gpu += 1