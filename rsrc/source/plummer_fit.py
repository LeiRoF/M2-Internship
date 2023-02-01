import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
import concurrent.futures
from matplotlib.lines import Line2D
from functions import simuini, loadfits, loadcloud

# Modified Plummer model
def mod_plummer(r, rhoflat, Rflat, etha):
    return rhoflat * ( Rflat / np.sqrt(Rflat**2 + r**2) )**etha

# Read pixel size from soc ini file
def read_pixsize(name):
    for line in open('ressource/{}_soc.ini'.format(name)).readlines():
        s = line.split('#')[0].split()
        key, a = s[0], s[1]

        if (key.find('gridlength')==0):
            return float(a)

# Make a radial gradient of the distances from the center corresponding either
# to the maximum of column density
# or to the maximum volume density
def radial(data, max_pos, ndim=False):
    indices = np.indices(data.shape)

    if ndim:
        indices[0] -= max_pos[0]
        indices[1] -= max_pos[1]
        indices[2] -= max_pos[2]

        radius = (indices[0]**2 + indices[1]**2 + indices[2]**2)**0.5       

    else:
        indices[0] -= max_pos[0]
        indices[1] -= max_pos[1]

        radius = (indices[0]**2 + indices[1]**2)**0.5

    return radius

# Make a list of mean value along a circle centered at the maximum of column density
def mean_list(radius, data, offset, max_pos, ndim=False):
    mean_list = np.zeros(offset)

    for i in range(1, offset):
        mask = (radius >= i) & (radius < i+1)
        mean_list[i] = np.mean(data[mask])

    if ndim:
        mean_list[0] = data[max_pos[0], max_pos[1], max_pos[2]]
    else:
        mean_list[0] = data[max_pos[0], max_pos[1]]

    return mean_list

# Fit
def fitting(x, y):
    popt, pcov = curve_fit(lambda x, rc, etha: mod_plummer(x, np.max(y), rc, etha), x, y, p0=[0.01, 4], bounds=([0, 1], [np.inf, 10]), maxfev=5000)

    rc = popt[0]
    etha = popt[1]

    src = np.sqrt(pcov[0, 0])
    setha = np.sqrt(pcov[1, 1])

    return rc, etha, src, setha

# Extract volume density profile
def true_prof3d(index, cloud):
    profile3d = cloud[index[0], index[1], index[2]]
    max1 = np.max(profile3d)
    return profile3d/max1

# Extract mean volume density profile
def mean_prof3d(cloud, offset, maxpos):
    radius = radial(cloud, maxpos, ndim=True)

    y = mean_list(radius, cloud, offset, maxpos, ndim=True)

    #y /= np.max(y)
    return y

# Extract mean column density profile
def mean_prof2d(data, offset, maxpos):
    radius = radial(data, maxpos)

    y = mean_list(radius, data, offset, maxpos)

    #y /= np.max(y)
    return y


# Extract true column density profile
def true_prof2d(index, data, diag=False):
    if diag:
        profile2d = data[index]
    else:
        profile2d = data[index[0], index[1]]
    profile2d /= np.max(profile2d)
    return profile2d

# Fit horizontal and vertical plummer profile
def fit_cd(data, x, index):
    y = data[index[0], index[1]]
    rc, etha, src, setha = fitting(x, y)

    fit = mod_plummer(x, np.max(y), rc, etha)
    maxv = np.max(fit)
    fit /= maxv
    return fit, rc, etha, src, setha

# Fit diagonal plummer profile
def fit_cd1(data, x, index):
    y = data[index]
    rc, etha, src, setha = fitting(x, y)

    fit = mod_plummer(x, np.max(y), rc, etha)
    maxv = np.max(fit)
    fit /= maxv
    return fit, rc, etha, src, setha

# Core function to fit single plummer profile on realisation maps
def uncertainty_fit_core(data, x, index, diag):
    if diag:
        y = data[index]
    else:
        y = data[index[0], index[1]]
    rc, etha, _, _ = fitting(x, y)
    fit = mod_plummer(x, np.max(y), rc, etha)
    fit /= np.max(fit)

    return (fit)

# Fit the mean profile
def fit_mean_profile(data, x, offset, max_pos):
    radius = radial(data, max_pos)
    y = mean_list(radius, data, offset, max_pos)

    rc, etha, src, setha = fitting(x, y)

    fit = mod_plummer(x, np.max(y), rc, etha)
    maxv = np.max(fit)
    #fit /= maxv
    return fit, rc, etha, src, setha

# Fit the mean plummer profile on realisation maps
def uncertainty_mean_profile(data, x, offset, max_pos):
    radius = radial(data, max_pos)
    y = mean_list(radius, data, offset, max_pos)

    rc, etha, _, _ = fitting(x, y)
    fit = mod_plummer(x, np.max(y), rc, etha)
    fit /= np.max(fit)

    return (fit)

# Main function to determine the uncertainty on each fitted plummer profile
def uncertainty_fit_main(data, x, offset, max_pos, index=None, diag=False, mean=False):
    outputfits = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=p['nbcpu']) as executor:
        if mean:
            for d in data:
                outputfits.append(executor.submit(uncertainty_mean_profile, d, x, offset, max_pos))
        else:
            for d in data:
                outputfits.append(executor.submit(uncertainty_fit_core, d, x, index, diag))      
    
    fit_array = np.zeros((len(data), len(x)))
    for i, o in enumerate(outputfits):
        fit_array[i] = o.result()          
    
    fitm = np.zeros(len(x))
    fitp = np.zeros(len(x))

    for i in range(len(x)):
        fitm[i] = np.percentile(fit_array[:, i], 16)
        fitp[i] = np.percentile(fit_array[:, i], 84)
    
    return fitm, fitp


# Function that extract and fit Plummer profile along a single profile
# and produce one part of the output plots
def multi_fit(dirfit, data, data1, datalist, sub, fig, pixs, offset, xm, ym, zm, max_pos, cloud, x1):
    test = []
    if dirfit == 0:
        fitrange = range(max_pos[0], max_pos[0]+offset)
        x = np.array([i*pixs for i in range(offset)])
        index = [fitrange, ym, zm]
        index1 = [fitrange, max_pos[1]]
        test = datalist
        arrow = [offset, 0]

    if dirfit == 90:
        fitrange = range(max_pos[1], max_pos[1]-offset, -1)
        x = np.array([i*pixs for i in range(offset)])
        index = [xm, fitrange, zm]
        index1 = [max_pos[0], fitrange]
        test = datalist
        arrow = [0, offset]

    if dirfit == 180:
        fitrange = range(max_pos[0], max_pos[0]-offset, -1)
        x = np.array([i*pixs for i in range(offset)])
        index = [fitrange, ym, zm]
        index1 = [fitrange, max_pos[1]]
        test = datalist
        arrow = [-offset, 0]

    if dirfit == 270:
        fitrange = range(max_pos[1], max_pos[1]+offset)
        x = np.array([i*pixs for i in range(offset)])
        index = [xm, fitrange, zm]
        index1 = [max_pos[0], fitrange]
        test = datalist
        arrow = [0, -offset]

    if dirfit == 45:
        data = np.flipud(data).diagonal()
        data1 = np.flipud(data1).diagonal()
        for i in range(p['uncertmc']):
            test.append(np.flipud(datalist[i]).diagonal())

        fitrange = range(max_pos[0], max_pos[0]+offset)
        fitrangey = range(max_pos[1], max_pos[1]-offset, -1)
        x = np.array([i*pixs for i in range(offset)])
        index = [fitrange, fitrangey, zm]
        arrow = [fact*offset, fact*offset]

    if dirfit == 135:
        data = np.diagonal(data)
        data1 = np.diagonal(data1)
        for i in range(p['uncertmc']):
            test.append(np.diagonal(datalist[i]))

        fitrange = range(max_pos[0], max_pos[0]-offset, -1)
        fitrangey = range(max_pos[1], max_pos[1]-offset, -1)
        x = np.array([i*pixs for i in range(offset)])
        index = [fitrange, fitrangey, zm]
        arrow = [-fact*offset, fact*offset]

    if dirfit == 225:
        data = np.fliplr(data).diagonal()
        data1 = np.fliplr(data1).diagonal()
        for i in range(p['uncertmc']):
            test.append(np.fliplr(datalist[i]).diagonal())

        fitrange = range(max_pos[1], max_pos[1]+offset)
        fitrangex = range(max_pos[0], max_pos[0]-offset, -1)
        x = np.array([i*pixs for i in range(offset)])
        index = [fitrangex, fitrange, zm]
        arrow = [-fact*offset, -fact*offset]
    
    if dirfit == 315:
        data = np.diagonal(data)
        data1 = np.diagonal(data1)
        for i in range(p['uncertmc']):
            test.append(np.diagonal(datalist[i]))

        fitrange = range(max_pos[0], max_pos[0]+offset)
        fitrangey = range(max_pos[1], max_pos[1]+offset)
        x = np.array([i*pixs for i in range(offset)])
        index = [fitrange, fitrange, zm]
        arrow = [fact*offset, -fact*offset]


    if (dirfit == 0) or (dirfit == 90) or (dirfit == 180) or (dirfit == 270):
        # True density profile
        t_profile3d = true_prof3d(index, cloud)
        t_profile2d = true_prof2d(index1, data)

        # Fit True cd
        fit, rc, etha, src, setha = fit_cd(data, x, index1)
        
        # Fit ff cd
        fit1, rc1, etha1, src1, setha1 = fit_cd(data1, x, index1)
        bornm1, bornp1 = uncertainty_fit_main(test, x, offset, max_pos, index=index1)
        profile2d = true_prof2d(index1, data1)
    
    else:
        # True density profile
        t_profile3d = true_prof3d(index, cloud)
        t_profile2d = true_prof2d(fitrange, data, diag=True)

        # Fit True cd
        fit, rc, etha, src, setha = fit_cd1(data, x, fitrange)
        
        # Fit ff cd
        fit1, rc1, etha1, src1, setha1 = fit_cd1(data1, x, fitrange)
        bornm1, bornp1 = uncertainty_fit_main(test, x, offset, max_pos, index=fitrange, diag=True)
        profile2d = true_prof2d(fitrange, data1, diag=True)

    
    # Uncertainties desabled
    # bornp = mod_plummer(x, np.max(data), rc+src, etha+setha)
    # bornp /= np.max(bornp)
    # bornm = mod_plummer(x, np.max(data), rc-src, etha-setha)
    # bornm /= np.max(bornm)

    # plot
    ax = fig.add_subplot(3, 3, sub)
    ax.plot(x, fit, label='fit_true', color='r')
    ax.plot(x, fit1, label='fit_ff', color='b')
    ax.plot(x, t_profile3d, label='true3d', color='g', ls=':')
    ax.plot(x, t_profile2d, label='true2d', color='r', ls='--')
    ax.plot(x, profile2d, label='true_ff', color='b', ls='--')

    # Uncertainties desabled
    # ax.fill_between(x,
    #                 bornp,
    #                 bornm,
    #                 alpha=0.1, color='r', lw=0.0)
    # ax.fill_between(x,
    #                 bornp1,
    #                 bornm1,
    #                 alpha=0.1, color='b', lw=0.0)   

# Main function
def fit_main(name, core):
    # Pixel size
    pixs = read_pixsize(name)

    # Column density
    data = dataplot = loadfits(name + '_cd_true')
    data1 = loadfits(name + '_cd_ff')

    # Position of the maximum
    max_pos = arrow_cent = np.unravel_index(data.argmax(), data.shape)

    # Maximum distance between the position of the maximum and the border of the map
    xm = min(data.shape[0] - 1 - max_pos[0], max_pos[0])
    ym = min(data.shape[1] - 1 - max_pos[1], max_pos[1])

    # Cloud
    cloud, N = loadcloud(name)
    cloud = cloud.swapaxes(0, 2) # (z, y, x) to (x, y, z)
    max_pos2 = np.unravel_index(cloud.argmax(), cloud.shape) 
    zm = N[0]//2
    offset = min(xm, ym, zm)

    if (max_pos[0] + offset) > np.shape(data)[0]-1:
        offset -= max_pos[0] + offset - np.shape(data)[0] +1
    if (max_pos[1] + offset) > np.shape(data)[1]-1:
        offset -= max_pos[1] + offset - np.shape(data)[1] +1

    if (max_pos[0] - offset) < 0:
        offset -= abs(max_pos[0] - offset)
    if (max_pos[1] - offset) < 0:
        offset -= abs(max_pos[1] + offset)  

    #Re-center data (for diagonal profile)
    data = data[max_pos[0]-offset:max_pos[0]+offset, max_pos[1]-offset:max_pos[1]+offset]
    data1 = data1[max_pos[0]-offset:max_pos[0]+offset, max_pos[1]-offset:max_pos[1]+offset]
    cloud = cloud[max_pos[0]-offset:max_pos[0]+offset, max_pos[1]-offset:max_pos[1]+offset, max_pos2[2]-offset:max_pos2[2]+offset]

    # Uncertainties maps
    datalist1 = []
    for i in range(p['uncertmc']):
        datalist1.append(loadfits('{}_cd_ff'.format(name)))#[max_pos[0]-offset:max_pos[0]+offset, max_pos[1]-offset:max_pos[1]+offset])

    max_pos = np.unravel_index(data.argmax(), data.shape)
    # max_pos = np.unravel_index(data1.argmax(), data1.shape)
    # max_pos2 = np.unravel_index(cloud.argmax(), cloud.shape) 

    # Fit and plot
    x1 = np.array([i*pixs for i in range(offset)])

    ################# DEBUT
    # plt.figure()
    # for i in range(p['uncertmc']):
    #     fit, _, _, _, _ = fit_mean_profile(datalist1[i], x1, offset, max_pos)
    #     plt.plot(x1, fit, c='gray')

    # fittrue, _, _, _, _ = fit_mean_profile(data1, x1, offset, max_pos)
    # plt.plot(x1, fittrue, c='r')
    # plt.show()

    # plt.figure()
    # for i in range(p['uncertmc']):
    #     radius = radial(datalist1[i], max_pos)
    #     y = mean_list(radius, datalist1[i], offset)
    #     plt.plot(x1, y, c='gray')

    # radius = radial(data1, max_pos)
    # y = mean_list(radius, data1, offset)
    # plt.plot(x1, y, c='r', lw=1.5, label='Obs')

    # radius = radial(data, max_pos)
    # y = mean_list(radius, data, offset)
    # plt.plot(x1, y, c='b', lw=1.5, label='true')
    # plt.legend()
    # #plt.savefig('output_plots/{}.pdf'.format(name))
    # plt.show()
    ################# FIN

    # Compute mean profile3ds
    fit2, rc2, etha2, src2, setha2 = fit_mean_profile(data, x1, offset, max_pos)
    fit3, rc3, etha3, src3, setha3 = fit_mean_profile(data1, x1, offset, max_pos)
    #bornm3, bornp3 = uncertainty_fit_main(datalist1, x1, offset, max_pos, mean=True)
    t_mean3d = mean_prof3d(cloud, offset, [*max_pos, zm])
    t_mean2d = mean_prof2d(data, offset, max_pos)
    mean2d = mean_prof2d(data1, offset, max_pos)

    # Uncertainties desabled
    # bornp2 = mod_plummer(x1, np.max(data), rc2+src2, etha2+setha2)
    # bornp2 /= np.max(bornp2)
    # bornm2 = mod_plummer(x1, np.max(data), rc2-src2, etha2-setha2)
    # bornm2 /= np.max(bornm2)

    # bornp3 = mod_plummer(x1, np.max(data1), rc3+src3, etha3+setha3)
    # bornp3 /= np.max(bornp3)
    # bornm3 = mod_plummer(x1, np.max(data1), rc3-src3, etha3-setha3)
    # bornm3 /= np.max(bornm3)

    fig = plt.figure(figsize=(8, 7))

    # Compute and plot profile3ds
    multi_fit(0, data, data1, datalist1, 2, fig, pixs, offset, xm, ym, zm, max_pos, cloud, x1)
    multi_fit(45, data, data1, datalist1, 9, fig, pixs, offset, xm, ym, zm, max_pos, cloud, x1)
    multi_fit(90, data, data1, datalist1, 4, fig, pixs, offset, xm, ym, zm, max_pos, cloud, x1)
    multi_fit(135, data, data1, datalist1, 7, fig, pixs, offset, xm, ym, zm, max_pos, cloud, x1)
    multi_fit(180, data, data1, datalist1, 8, fig, pixs, offset, xm, ym, zm, max_pos, cloud, x1)
    multi_fit(225, data, data1, datalist1, 1, fig, pixs, offset, xm, ym, zm, max_pos, cloud, x1)
    multi_fit(270, data, data1, datalist1, 6, fig, pixs, offset, xm, ym, zm, max_pos, cloud, x1)
    multi_fit(315, data, data1, datalist1, 3, fig, pixs, offset, xm, ym, zm, max_pos, cloud, x1)

    # Pixel to pc
    offset *= pixs
    shape = dataplot.shape[0]
    xmax = (shape - arrow_cent[1]) * pixs
    xmin = -(arrow_cent[1]) * pixs
    ymax = (shape - arrow_cent[0]) * pixs
    ymin = -(arrow_cent[0]) * pixs

    # Arrows size
    s = offset*0.003
    hs = s*30

    ax = fig.add_subplot(335)

    # Plot column density map
    im = ax.imshow(dataplot, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='CMRmap')

    # Plot arrows
    ax.arrow(0, 0, offset, 0, width=s, head_width=hs, length_includes_head=True)
    ax.arrow(0, 0, 0, offset, width=s, head_width=hs, length_includes_head=True)
    ax.arrow(0, 0, -offset, 0, width=s, head_width=hs, length_includes_head=True)
    ax.arrow(0, 0, 0, -offset, width=s, head_width=hs, length_includes_head=True)
    ax.arrow(0, 0, -fact*offset, fact*offset, width=s, head_width=hs, length_includes_head=True)
    ax.arrow(0, 0, fact*offset, -fact*offset, width=s, head_width=hs, length_includes_head=True)
    ax.arrow(0, 0, -fact*offset, -fact*offset, width=s, head_width=hs, length_includes_head=True)
    ax.arrow(0, 0, fact*offset, fact*offset, width=s, head_width=hs, length_includes_head=True)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Legend
    lines = [Line2D([0], [0], color='g', ls=':', lw=1),
            Line2D([0], [0], color='r', ls='--', lw=1),
            Line2D([0], [0], color='r', lw=1),
            Line2D([0], [0], color='b', ls='--', lw=1),
            Line2D([0], [0], color='b', lw=1)]
    labels = [r' Simu 3D', r' Simu 2D', r' Simu fit', r' Obs 2D', r' Obs fit']
    
    fig.legend(lines, labels, loc = [0.09, 0.93], ncol=5)
    fig.text(0.48, 0.05, horizontalalignment='center', s=r'Distance (pc)')
    fig.text(0.05, 0.5, verticalalignment='center', s=r'Normalized density', rotation=90)

    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, shrink=0.75, label=r'Column density ($H_2.cm^{-2}$)')

    fig.savefig('output_plots/{}_plummer.pdf'.format(name), bbox_inches='tight')
    plt.close() 

    # Mean profiles plot
    plt.figure(figsize=(3, 3))
    #plt.plot(x1, t_mean3d/np.max(t_mean3d) * np.max(t_mean2d), label=r' Simu 3D', color='g', ls=':')
    plt.plot(x1, t_mean2d, label=r' Simu 2D', color='r', ls='--')
    plt.plot(x1, fit2, label=r' Simu fit', color='r')
    plt.plot(x1, mean2d, label=r' Obs 2D', color='b', ls='--')
    plt.plot(x1, fit3, label=r' Obs fit', color='b')

    # Uncertainties desabled
    # plt.fill_between(x1,
    #             bornp2,
    #             bornm2,
    #             alpha=0.1, color='orange', lw=0.0)
    # plt.fill_between(x1,
    #             bornp3,
    #             bornm3,
    #             alpha=0.1, color='magenta', lw=0.0) 

    plt.legend()
    plt.xlabel(r'Distance (pc)')
    plt.ylabel(r'Column density ($H_{2}.cm^{-2}$)')
    plt.savefig('output_plots/{}_plummer_means.pdf'.format(name), bbox_inches='tight')

    print('Core {} done'.format(core))

    return [core, offset, data1[max_pos[0], max_pos[1]], rc3, etha3, src3, setha3]

if __name__ == '__main__':

    matplotlib.use('agg')  # Prevent plotting error with multiprocesses

    fact = np.sqrt(2)/2

    # Parameters
    p = simuini()

    print('-------------------------------')
    print('          Plummer fit          ')
    print('-------------------------------')

    out = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=p['nbthread']*p['nbsocket']) as executor:
        for c in range(1, p['nbcore']+1):
            out.append(executor.submit(fit_main, '{}/{}core{}/{}core{}'.format(p['name'], p['name'], c, p['name'], c), c))
    
    para_array = np.zeros((p['nbcore'], 7))
    for i, o in enumerate(out):
        res = o.result()
        j = res[0] - 1
        para_array[j] = res

    # Save the fitted parameters of the mean plummer profile of each core
    np.savetxt('ressource/{}/{}_plummer_param.dat'.format(p['name'], p['name']), para_array)