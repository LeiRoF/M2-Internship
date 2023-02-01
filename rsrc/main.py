import os
import sys
import subprocess
sys.path.append('source')
from functions import simuini

def gpu_runlist(p):
    """
    Function that return a list that allows to distribute SOC and LOC tasks
    on mulitple GPUs

    Arguments:
    ----------
        p: Dictionnary 
            Dictionnary containing the value of each parameter

    Return:
    -------
        outlist: list
            List containing number of task to run simultaneously

    """

    nbcore = p['nbcore']
    increment = p['nbgpu']

    outlist = []

    while nbcore > 0:
        if nbcore-increment < 0:
            increment += (nbcore-increment)
        nbcore -= increment
        outlist.append(increment)
    
    return outlist

def command(name, i, locsoc, mol=1):
    """
    Function that return a command to lunch either SOC or LOC

    Arguments:
    ----------
        name: str
            Name of the cloud from which the object of interest was extracted

        i: int
            Number of the core

        locsoc: str
            Choose between a SOC or a LOC run

    Return:
    -------
        str
            Command to run SOC or LOC in a shell
    """

    name = '{}core{}'.format(p['name'], i)
    folder = '{}/{}'.format(p['name'], name)
    if locsoc == 'soc':
        return 'python3 external/SOC/SOC_master/ASOC.py ressource/{}/{}_soc.ini'.format(folder, name)
    if locsoc == 'loc':
        return 'python3 external/LOC/LOC_master/LOC_OT.py ressource/{}/{}_{}_loc.ini'.format(folder, name, mol)

def runlocsoc(locsoc, in_list, p):
    """
    Function that launch SOC or LOC for an object

    Arguments:
    ----------
        locsoc: str
            Choose between a SOC or a LOC run
        
        in_list: list
            List containing number of task to run simultaneously

        p: Dictionnary 
            Dictionnary containing the value of each parameter
    """
    if locsoc =='soc':
        core = 1
        for elem in in_list:
            main_command = ''
            for i in range(elem-1):
                main_command += command(p['name'], core, locsoc)
                main_command += ' & '
                core += 1
            main_command += command(p['name'], core, locsoc)
            #print(main_command)
            core += 1
            subprocess.run(main_command, shell=True)
    else:
        for mol in p['molecule']:
            core = 1
            for elem in in_list:
                main_command = ''
                for i in range(elem-1):
                    main_command += command(p['name'], core, locsoc, mol)
                    main_command += ' & '
                    core += 1
                main_command += command(p['name'], core, locsoc, mol)
                #print(main_command)
                core += 1
                subprocess.run(main_command, shell=True)

# Read parameters
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
for n in range(p['nbcore']):
    name = '{}core{}'.format(p['name'], n+1)
    folder = '{}/{}'.format(p['name'], name)
    
    if os.path.exists('ressource/{}'.format(folder)) == False:
        os.mkdir('ressource/{}'.format(folder))
    if os.path.exists('output_files/{}'.format(folder)) == False:
        os.mkdir('output_files/{}'.format(folder))
    if os.path.exists('output_plots/{}'.format(folder)) == False:
        os.mkdir('output_plots/{}'.format(folder))

os.environ['OMP_NUM_THREADS'] = str(p['nbthread'])

if (p['core'] == 2) or (p['core'] == 1):
    print("------------------------------")
    print("       Core extraction        ")
    print("------------------------------")
    
    subprocess.run('mpirun -n 1 --map-by numa:pe={} --bind-to none python3 source/core_only.py'.format(p['nbcpu']), shell=True)

if p['core'] == 3:
    print("------------------------------")
    print("           AMUSE run          ")
    print("------------------------------")

    subprocess.run('mpirun -n 1 --map-by numa:pe={} --bind-to none python3 source/cloud_and_core.py'.format(p['nbcpu']), shell=True)

runlist = gpu_runlist(p)

if p['soc']:

    print("------------------------------")
    print("           SOC run            ")
    print("------------------------------")

    runlocsoc('soc', runlist, p)

    # Remove unecessary files
    for i in range(1, p['nbcore']+1):
        folder = 'ressource/{}/{}core{}/'.format(p['name'], p['name'], i)
        os.system('rm {}/*_emitted.data'.format(folder))
    os.system('rm packet.info')

    print("------------------------------")
    print("              Fit             ")
    print("------------------------------")

    subprocess.run('python3 source/fit_map.py', shell=True)
    #subprocess.run('python3 source/fit_map_uncert.py', shell=True)

    subprocess.run('python3 source/plummer_fit.py', shell=True)


if p['loc']:

    print("------------------------------")
    print("       Making LOC cloud       ")
    print("------------------------------")

    subprocess.run('python3 source/loc_cloud.py', shell=True)

    print("------------------------------")
    print("           LOC run            ")
    print("------------------------------")   

    runlocsoc('loc', runlist, p)

    # Remove unecessary files
    for i in range(1, p['nbcore']+1):
        folder = 'ressource/{}/{}core{}/'.format(p['name'], p['name'], i)
        os.system('rm {}*.save'.format(folder))
    os.system('rm *.dump')
    os.system('rm gauss_py.dat')

    print("------------------------------")
    print("           Spectra            ")
    print("------------------------------")   

    subprocess.run('python3 source/spectra_uncert.py', shell=True)
    subprocess.run('python3 source/spectra.py', shell=True)

print("------------------------------")
print("    Computation completed     ")
print("------------------------------")
