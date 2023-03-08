import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy.io import fits
from functions import simuini, loadcloud, loadtemp, units, constants
from scipy.optimize import curve_fit
import concurrent.futures

# Black body
def bb(l, T):
    """
    Black body law

    Arguments:
        l: float
            Wavelenght
        
        T: float
            Temperature
    
    Return:
    -------
        float
            The result of the black body law according to the arguments
    """

    return ((2*h*c**2)/(l**5)) * (1/(np.exp((h*c) / (l*kb*T)) - 1))

# Modified Black Body
def mbb(l, T, N):
    """
    Modified black body law

    Arguments:
        l: float
            Wavelenght
        
        T: float
            Temperature

        N: float
            Column density
    
    Return:
    -------
        float
            The result of the modified black body law according to the arguments
    """

    return bb(l, T) * k0 * (l0/l)**2.0 * mmh * N

def jy2si(a, f):
    """
    Convert Jy/sr to W.m²/m/sr

    Arguments:
    ---------
        a: value to convert

        f: frequency

    Return:
        float
            Result of the conversion

    """
    return a * 10**-26 * f**2 / c

def loadmaps(wavel, name, N):
    """
    Function that loads mulitple fits maps

    Arguments:
        wavel: numpy array
            list of wavelenghts
        
        name: name of the files without the wavelenghts information

        N: Resolutions of the map

    Return:
        out: numpy array
            Array containing the different maps
    """

    nwavel = len(wavel)
    out = np.zeros((nwavel, N[1], N[2]))

    for i in range(nwavel):
        fitmap = fits.open('output_files/{}_{}.fits'.format(name, wavel[i]))
        out[i, :, :] = fitmap[0].data.astype(np.float64)
        fitmap.close()

    return out

def fitter(data, wavel):
    """
    Function that construct temperature map from surface brightness by fitting

    Arguments:
    ----------
        data: numpy array
            surface brightness maps that will be used for the fit

        wavel: numpy array
            List of wavelenghts

    Returns:
        T: numpy array
            Temperature map obtain from the fit

        N: numpy array
            Column density map obtain from the fit
    """
    dims = np.shape(data)
    # Temperature map
    T = np.zeros((dims[1], dims[2]))
    # Column density map
    N = np.zeros((dims[1], dims[2]))

    # Convert wavelenghts to m
    wavel = wavel*1e-6

    # Wavelenghts to frequencies
    freq = c/wavel

    for y in range(dims[1]):
        for x in range(dims[2]):
            # Keep only one pixel
            data_temp = jy2si(data[0:len(freq), y, x], freq)

            # Fit
            popt, pcov = curve_fit(mbb, wavel, data_temp, p0 = [10.0, 1e24])
            #err = np.sqrt(np.diag(pcov))

            # y, x, order to x, y for convenience
            T[x, y] = popt[0]
            N[x, y] = popt[1]*1e-4  # m-2 to cm-2

    return T, N    

def T_dust(T, cloud, N):
    """
    Function that will return the mass weighted mean temperature map and
    a minimal temperature map of an object

    Arguments:
    ----------
        T: numpy array
            Temperature cube

        cloud: numpy array
            Density cube

        N: Resolution of the cubes

    Returns:
    -------
        temp: numpy array
            Mass weighted mean temperature map

        temp_min: numy array
            Minimal temperature map

    """
    temp = np.zeros((N[2], N[1]))
    temp_min = np.zeros((N[2], N[1]))

    for x in range(N[2]):
        for y in range(N[1]):
            # x, y swap for convenience
            M = sum(cloud[:, y, x])
            temp[x, y] = sum((cloud[:, y, x]/M)*T[:, y, x])
            temp_min[x, y] = np.min(T[:, y, x])

    return temp, temp_min

def loadini(name, N):
    """
    
    """

    for line in open('ressource/{}_soc.ini'.format(name)).readlines():
        s = line.split('#')[0].split()
        key, a  =  s[0], s[1]
        if (key.find('gridlen')==0):
            gridlen = float(a)
            return N[0] * gridlen * 0.5

def savefits(array, name):
    hdu = fits.PrimaryHDU(array)
    hdu.writeto('output_files/{}_cd_ff.fits'.format(name), overwrite=True)

def fit_main(name, n):
        fname = '{}core{}'.format(name, n)
        folder = '{}/{}'.format(name, fname)
        fpath = '{}/{}'.format(folder, fname)

        print(fpath)

        # Load cloud file
        cloud, N = loadcloud(fpath)

        # Load fits maps
        data = loadmaps(wavel, fpath, N)

        # Load SOC ini file for real cube size (in pc)
        csize = loadini(fpath, N)

        limit = [-csize, csize, -csize, csize]


        # Compute Temperature map
        Tmap, Nmap = fitter(data, wavel)

        # Show fitted column density
        plt.figure(figsize=(5,3))
        plt.imshow(Nmap, extent=limit, origin='lower', cmap='CMRmap')
        plt.colorbar(label=r'Column density ($H_2.cm^{-2}$)', shrink=1.0)
        plt.xlabel(r'Distance (pc)')
        plt.ylabel(r'Distance (pc)')
        plt.savefig('output_plots/{}_column_density_ff.pdf'.format(fpath), bbox_inches='tight')
        savefits(Nmap, fpath)
        print('Core {} column density computed'.format(n))

        # True dust temperature computation
        T = loadtemp(fpath)

        T_mean, T_min = T_dust(T, cloud, N)

        # Show Temperature maps
        plt.figure(figsize=(13, 10))
        plt.subplot(221)
        plt.imshow(Tmap, extent = limit, origin='lower', cmap='CMRmap')
        plt.title("Fitted dust temperature")
        plt.xlabel(r'Distance (pc)')
        plt.ylabel(r'Distance (pc)')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(r'$T_{\rm dust}$ (K)')

        plt.subplot(222)
        plt.imshow(T_mean, extent = limit, origin='lower', cmap='CMRmap')                    
        plt.title("Mean dust temperature")
        plt.xlabel(r'Distance (pc)')
        plt.ylabel(r'Distance (pc)')
        cbar1 = plt.colorbar()
        cbar1.ax.set_ylabel(r'$T_{\rm dust}$ (K)')

        plt.subplot(223)
        plt.imshow(Tmap - T_mean, extent = limit, origin='lower', cmap='CMRmap')
        plt.title("Dust temperature differences")
        plt.xlabel(r'Distance (pc)')
        plt.ylabel(r'Distance (pc)')
        cbar1 = plt.colorbar()
        cbar1.ax.set_ylabel(r'$\Delta T_{\rm dust}$ (K)')

        plt.subplot(224)
        plt.imshow(T_min, extent = limit, origin='lower', cmap='CMRmap')                    
        plt.title("Min dust temperature")
        plt.xlabel(r'Distance (pc)')
        plt.ylabel(r'Distance (pc)')
        cbar1 = plt.colorbar()
        cbar1.ax.set_ylabel(r'$T_{\rm dust}$ (K)')
        plt.savefig('output_plots/{}_Tdustmap_ff.pdf'.format(fpath), bbox_inches='tight')

        print('Core {} temperature maps done'.format(n))


if __name__ == '__main__':

    matplotlib.use('agg')  # Prevent plotting error with multiprocesses

    # Parameters
    p = simuini()

    # Constant
    h = constants.h.value_in(units.J * units.s)
    c = constants.c.value_in(units.m * units.s**-1)
    kb = constants.kB.value_in(units.J * units.K**-1)
    mp = 1.6726219e-27
    mmh = mp*p['mu']

    # Lambda0
    l0 = c/1000e9

    # Dust opacity at lambda0
    k0 = 0.01

    # List of wavelenghts (um)
    wavel = np.array([250, 350, 500])

    # dustfile = np.genfromtxt('ressource/aSilx.dust', skip_header=4)
    # f = dustfile[:, 0]
    # d = np.abs(f - 1000e9)
    # ind = np.argmin(d)
    # print(dustfile[ind, 2])
    # print((k0 * mmh)/(dustfile[ind, 2] * (1e-4 * 1e-2)**2 * np.pi))

    print('-------------------------------')
    print('            SED fit            ')
    print('-------------------------------')
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=p['nbthread']*p['nbsocket']) as executor:
        for core in range(1, p['nbcore']+1):
            executor.submit(fit_main, p['name'], core)
    