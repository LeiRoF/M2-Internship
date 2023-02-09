import numpy as np
import os
# from LRFutils import progress
from multiprocessing import Pool

# Function definition ---------------------------------------------------------

def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

def plummer(r, a):
    return 3/(4*np.pi*a**3)*(1 + r**2 / a**2)**(-5/2)


def generate_one(save_as="data.npy"):
    print(f"Generating {save_as}...")

    # Config ----------------------------------------------------------------------

    N = 64 # resolution in pixel
    space_range = [-1, 1] # space interval (arbitrary unit)

    # Initializations -------------------------------------------------------------

    r, dr = np.linspace(space_range[0], space_range[1], N, endpoint=True, retstep=True)
    X, Y, Z = np.meshgrid(r, r, r)

    #==============================================================================
    # Dennsity generation
    #==============================================================================

    # Generating lorentzian in a bigger space -------------------------------------

    tmp_r, tmp_dr = np.linspace(space_range[0]*2, space_range[1]*2, N*2, endpoint=True, retstep=True)
    tmp_X, tmp_Y, tmp_Z = np.meshgrid(tmp_r, tmp_r, tmp_r)

    cloud = 1/(1+(tmp_X**2+tmp_Y**2+tmp_Z**2)**2*1000)

    # Adding noise ----------------------------------------------------------------

    cloud *= np.random.normal(0, 0.1, size=(N*2, N*2, N*2))

    # Computing the fourier transform ---------------------------------------------
    # and cropping to a smaller space 
    # in order to avoid symmetries induced by the lorentzian shape

    cloud = np.abs(np.fft.fftn(cloud))
    cloud = cloud[:N, :N, :N]

    # Applying Plummer profile ----------------------------------------------------

    a = 0.9
    p = plummer(np.sqrt(X**2+Y**2+Z**2), a)

    cloud *= p
    cloud = cloud / np.amax(cloud) # normalizing

    #==============================================================================
    # Velocity generation
    #==============================================================================

    # Computing density gradient --------------------------------------------------

    cloud_gradient = np.gradient(cloud, dr)

    vx_grad = cloud_gradient[0] / np.amax(np.abs(cloud_gradient[0]))
    vy_grad = cloud_gradient[1] / np.amax(np.abs(cloud_gradient[1]))
    vz_grad = cloud_gradient[2] / np.amax(np.abs(cloud_gradient[2]))

    # Removing aberrant values at the edges ---------------------------------------

    for v in [vx_grad, vy_grad, vz_grad]:
        v[0,:,:] = 0
        v[-1,:,:] = 0
        v[:,0,:] = 0
        v[:,-1,:] = 0
        v[:,:,0] = 0
        v[:,:,-1] = 0

    # Computing rotation ----------------------------------------------------------

    vx_rot = -Y / np.amax(np.abs(Y))
    vy_rot = X  / np.amax(np.abs(X))
    vz_rot = X * 0

    # Computing the mean of the two velocity fields -------------------------------

    vx = (vx_grad + vx_rot)/2
    vy = (vy_grad + vy_rot)/2
    vz = (vz_grad + vz_rot)/2

    #==============================================================================
    # Spectrum generation
    #==============================================================================

    # Gaussian profile ------------------------------------------------------------

    f = np.linspace(0, 2, 100) # frequency range

    # def lorentzian(x, mu, gamma):
    #     return gamma / np.pi / ((x - mu)**2 + gamma**2)

    # def voigt(x, mu, sigma, gamma):
    #     return np.real(np.fft.ifft(np.fft.fft(gaussian(x, mu, sigma)) * np.fft.fft(lorentzian(x, mu, gamma))))

    gauss_profile = gaussian(f, 1, 0.1)

    # Hypercube generation --------------------------------------------------------

    spectrum_hypercube_x = np.zeros((N, N, N, 100))
    # spectrum_hypercube_y = np.zeros((N, N, N, 100))
    # spectrum_hypercube_z = np.zeros((N, N, N, 100))

    # Red Shift for each observation axis -----------------------------------------

    for i in range(N):
        for j in range(N):
            for k in range(N):
                spectrum_hypercube_x[i,j,k,:] = gaussian(f+3*vx[i,j,k], 1, 0.3) * cloud[i,j,k]
                # spectrum_hypercube_y[i,j,k,:] = gaussian(f+3*vy[i,j,k], 1, 0.3) * cloud[i,j,k]
                # spectrum_hypercube_z[i,j,k,:] = gaussian(f+3*vz[i,j,k], 1, 0.3) * cloud[i,j,k]

    # Stacking to 2D observation --------------------------------------------------

    spectrum_hypercube_x = np.sum(spectrum_hypercube_x, axis=0)
    # spectrum_hypercube_y = np.sum(spectrum_hypercube_y, axis=1)
    # spectrum_hypercube_z = np.sum(spectrum_hypercube_z, axis=2)

    # Keeping only 3 frequencies --------------------------------------------------

    X_data_cube = np.zeros((N, N, 3))
    X_data_cube[:,:,0] = spectrum_hypercube_x[:,:,25]
    X_data_cube[:,:,1] = spectrum_hypercube_x[:,:,50]
    X_data_cube[:,:,2] = spectrum_hypercube_x[:,:,75]

    # Y_data_cube = np.zeros((N, N, 3))
    # Y_data_cube[:,:,0] = spectrum_hypercube_y[:,:,25]
    # Y_data_cube[:,:,1] = spectrum_hypercube_y[:,:,50]
    # Y_data_cube[:,:,2] = spectrum_hypercube_y[:,:,75]

    # Z_data_cube = np.zeros((N, N, 3))
    # Z_data_cube[:,:,0] = spectrum_hypercube_z[:,:,25]
    # Z_data_cube[:,:,1] = spectrum_hypercube_z[:,:,50]
    # Z_data_cube[:,:,2] = spectrum_hypercube_z[:,:,75]

    #==============================================================================
    # Saving data
    #==============================================================================

    # Observation along X axis ----------------------------------------------------

    if not os.path.exists("./dataset/all"):
        os.makedirs("./dataset/all")

    np.savez_compressed(f"./dataset/all/{save_as}",
        rho_datacube=cloud,
        vx_datacube=vx,
        vy_datacube=vy,
        vz_datacube=vz,
        obs_datacube=spectrum_hypercube_x
    )

if __name__ == "__main__":
    N = 1000
    # bar = progress.Bar(1000, prefix="Generating dataset")

    r = os.getenv("OAR_NODEFILE")
    with open(r, 'r') as f:
        ncpu = len(f.readlines())

    p = Pool(ncpu)

    for i in range(N):
        # bar(i+1)
        
        p.apply_async(generate_one, args=(f"vector_{i}",))
    
    p.close()
    p.join()