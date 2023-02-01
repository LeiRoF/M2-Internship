import numpy as np
import concurrent.futures
from functions import simuini, savefits, load_spectra_3d, save_spectra_3d

def one_uncertainty_spectra(i, mol, t, p):
    np.random.seed()
    name = '{}/{}core{}/{}core{}'.format(p['name'], p['name'], i, p['name'], i)

    V, spe = load_spectra_3d('output_files/{}_{}_0{}-0{}.spe'.format(name, p['molecule'][mol], *t))
    NY, NX, NCHN = spe.shape

    max_spe = np.max(spe)

    for i in range(NX):
        for j in range(NY):
            spe[j, i, :] += np.random.normal(0, max_spe*0.05, NCHN)

    save_spectra_3d(V, spe, 'output_files/{}_{}_0{}-0{}.unspe'.format(name, p['molecule'][mol], *t))

def uncertainty_spectra(i, p):
    for mol in range(len(p['molecule'])):
        transition = p['transition'][mol].split(',')
        for tran in transition:
            t = tran.split('-')
            one_uncertainty_spectra(i, mol, t, p)
    
    print('Core {} done'.format(i))

if __name__ == '__main__':

    # Parameters
    p = simuini()

    with concurrent.futures.ProcessPoolExecutor(max_workers=p['nbthread']*p['nbsocket']) as executor:
        for core in range(1, p['nbcore']+1):
            executor.submit(uncertainty_spectra, core, p)