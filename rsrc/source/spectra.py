import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import concurrent.futures
from functions import simuini, loadfits, load_spectra_3d, load_spectra_3d_uncert
 

def read_pixsize(name):
    for line in open('ressource/{}_soc.ini'.format(name)).readlines():
        s = line.split('#')[0].split()
        key, a = s[0], s[1]

        if (key.find('gridlength')==0):
            return float(a)                                                                             

def plot_spectra(i, p):
    """
    Function that plot spectral map on top of a maximum density map

    Argument:
    ---------
        i: int
            Number of the core

        p: Dictionnary 
            Dictionnary containing the value of each parameter
    """

    name = '{}/{}core{}/{}core{}'.format(p['name'], p['name'], i, p['name'], i)

    pixs = read_pixsize(name)

    for mol in range(len(p['molecule'])):
        transition = p['transition'][mol].split(',')
        for tran in transition:
            t = tran.split('-')

            V, spe = load_spectra_3d_uncert('output_files/{}_{}_0{}-0{}.unspe'.format(name, p['molecule'][mol], *t))
            #V, T21 = LOC_read_spectra_3D('output_files/{}_{}_02-01.spe'.format(name, p['molecule']))
            NY, NX, NCHN = spe.shape  # number of pixels and channels

            print("Spectra file load")

            # One spectrum from (about) the model centre
            fig = plt.figure(figsize=(8, 7))

            imf = loadfits('{}_cd_true'.format(name))
            
            shape = imf.shape[0]
            maxpos = np.unravel_index(imf.argmax(), imf.shape)
            xmax = (shape - maxpos[1]) * pixs
            xmin = -(maxpos[1]) * pixs
            ymax = (shape - maxpos[0]) * pixs
            ymin = -(maxpos[0]) * pixs

            ax = fig.add_axes([0.0, 0.075, 0.85, 0.85])
            im = ax.imshow(imf, zorder=0, origin='lower', cmap='CMRmap',
            extent=[xmin, xmax, ymin, ymax])
            ax.tick_params(which='both', direction='out', bottom=False,
            top=True, left=False, right=True, labelbottom=False, labeltop=True,
            labelleft=False, labelright=True)

            axes = fig.subplots(nrows=p['resloc'], ncols=p['resloc'],
            sharex=True, sharey=True)
            index = 1
            for i in range(p['resloc']):
                for j in range(p['resloc']):
                    ax = axes[i, j]
                    ax.plot(V, spe[i, j, :], 'lime', zorder=1)
                    ax.axes.set_ylim(top=1.1*np.max(spe), bottom=np.min(spe))

                    if i == p['resloc'] -1 and j == 0:
                        ax.tick_params(which='both', direction='out',
                        bottom=True, top=False, right=False, left=True)
                        ax.set_xticks([-2, 0, 2])
                        #ax.spines['bottom'].set_color('k')
                    elif i == p['resloc'] -1:
                        ax.tick_params(which='both', direction='out',
                        bottom=True, top=False, right=False, left=False)
                        ax.set_xticks([-2, 0, 2])

                    elif j == 0:
                        ax.tick_params(which='both', direction='out',
                        bottom=False, top=False, right=False, left=True)
                    else:
                        ax.xaxis.set_visible(False)
                        ax.yaxis.set_visible(False)

                    ax.patch.set_alpha(0.0)
                    index += 1

            plt.subplots_adjust(0.054, 0.075, 0.796, 0.925, 0, 0)

            fig.text(0.44, 0.01, horizontalalignment='center', s=r'Velocity (km/s)')
            fig.text(0.005, 0.5, verticalalignment='center', s=r'$T_{mb}$ (K)', rotation=90)
            fig.text(0.44, 0.97, horizontalalignment='center', s=r'Distance (pc)')
            fig.text(0.855, 0.5, verticalalignment='center', s=r'Distance (pc)', rotation=90)

            #fig.subplots_adjust(right=0.83)
            cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
            fig.colorbar(im, cax=cbar_ax, shrink=0.75, label=r'Column density ($H_2.cm^{-2}$)')
            fig.savefig('output_plots/{}_{}_{}-{}_spectra.pdf'.format(name, p['molecule'][mol], *t))


    print("Spectra done")

if __name__ == '__main__':

    matplotlib.use('agg')  # Prevent plotting error with multiprocesses

    # Parameters
    p = simuini()

    with concurrent.futures.ProcessPoolExecutor(max_workers=p['nbthread']*p['nbsocket']) as executor:
        for core in range(1, p['nbcore']+1):
            executor.submit(plot_spectra, core, p)
    #plot_spectra(1, p)