echo "Installing AMUSE dependencies"

sudo apt-get install build-essential gfortran python3-dev \
  libopenmpi-dev openmpi-bin \
  libgsl-dev cmake libfftw3-3 libfftw3-dev \
  libgmp3-dev libmpfr6 libmpfr-dev \
  libhdf5-serial-dev hdf5-tools \
  libblas-dev liblapack-dev \
  python3-venv python3-pip git


pip install --upgrade pip
pip install numpy docutils mpi4py h5py wheel

pip install scipy astropy jupyter pandas seaborn matplotlib
pip install amuse-framework
pip install amuse-fi


echo "Installing SOC and LOC dependencies"

sudo apt-get install pocl-opencl-icd python3-pyopencl
sudo apt install ocl-icd-libopencl1
sudo apt install opencl-headers
sudo apt install clinfo

echo "Installing further dependencies"

pip install pyvtk astrodendro

echo "Installation done"