"""
DATASET CLASS
-------------

A dataset is composed of two dictionnaries of numpy ndarrays:
- x: inputs
- y: outputs

Each entry (called 'fields') of the dictionaries correspond to a different data type (e.g. cat image, 3D car model, mass of a planet, etc.).

The first dimension of the ndarrays correspond to the vector number.
Thus, the size of the first dimension corresponds to the number of vectors in the dataset.
It must be the same for all the inputs and outputs.

The next dimensions are free, depending on the field's data type.
However, it must be consistent for a given field.
"""

import numpy as np
from typing import Union, Callable
import os
from LRFutils import progress, logs
from . import sysinfo
import matplotlib.pyplot as plt
from copy import deepcopy as copy
from multiprocessing import Pool

class Vector():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            try:
                kwargs[key] = np.array(value)
            except:
                raise TypeError(f"{key} must be convertible to a numpy array")
        self.data = kwargs

class Dataset():

    def __init__(self,
                 name:str='Unammed',
                 loader:"function"=None,
                 raw_path:Union[str,None]=None,
                 archive_path:Union[str,None]=None,
                 val_frac:float=0.2,
                 test_frac:float=0.1,
                 autosave:bool=True,
                 verbose:bool=False,
                ) -> "Dataset":
        """
        Create a user-friendly dataset.
        - loader & raw path allow to load raw data
        - If archive_path is given and the file exist, it will ignore the raw path.
        """

        # Attribute definition
        self.name:str = name
        self.archive_path:str = archive_path
        self.loader:function = loader
        self.raw_path:str = raw_path
        self.archive_path:str = archive_path

        if 0 > val_frac + test_frac >= 1:
            raise ValueError("val_frac, test_frac and the sum of both must be between 0 and 1.")

        self.val_frac:float = val_frac
        self.test_frac:float = test_frac

        self.data:dict[np.ndarray] = {}
        self.means:dict[float] = {}
        self.stds:dict[float] = {}
        self.mins:dict[float] = {}
        self.maxs:dict[float] = {}

        self.xlabels:list[str] = None
        self.ylabels:list[str] = None

        if archive_path and os.path.isfile(archive_path):
            logs.warn("Archive loading is not implemented yet. Ingoring archive path.")   
            # self.load_archive()
        # else:
        # {
        #    if archive_path:
        #        logs.info("No archive found. Loading raw data.")
        if not (loader and raw_path):
            raise ValueError("To create a dataset from raw data, both loader and raw_path must be given.")
        if not callable(loader):
            raise TypeError('"loader" must be a callable object')
        if not os.path.isdir(raw_path):
            raise ValueError(f"Raw data path {raw_path} is not a valid directory.")
        self._load_raw(verbose=verbose)
        # }

        self.process(verbose=verbose)
            
        if autosave and archive_path is not None:
            logs.warn("Save archive is not implemented yet.")
            # self.save()

        if verbose:
            logs.info(f"Dataset created ✅\n{self}")

    # Save & load dataset in numpy archive ------------------------------------

    # TODO

    # def _load_archive(self, verbose=True):
    #     """Load data from archive data path"""

    #     if verbose:
    #         logs.info(f"Loading {self.name} dataset data from archive {self.archive_path}...")

    #     archive = np.load(self.archive_path, allow_pickle=True)

    #     for key, value in archive.items():
    #         if key.startswith('x_'):
    #             self.x[key[2:]] = value
    #         elif key.startswith('y_'):
    #             self.y[key[2:]] = value

    #     if verbose:
    #         logs.info(f"Loaded {self.name} dataset from archive ✅")

    #     return self

    # def save(self):
    #     """Save the dataset to a numpy compressed file"""

    #     kwargs = {}
    #     for key, value in self.x.items():
    #         kwargs[f'x_{key}'] = value
    #     for key, value in self.y.items():
    #         kwargs[f'y_{key}'] = value

    #     np.savez_compressed(self.archive_path, **kwargs)
    
    # Load from raw data ------------------------------------------------------
    
    def _load_raw(self, verbose=True):
        """Load data from raw data path"""

        files = os.listdir(self.raw_path)

        if verbose:
            logs.info(f"Loading {self.name}'s raw data from {self.raw_path}...")
            bar = progress.Bar(len(files))

        for i, item in enumerate(files):

            vector = self.loader(os.path.join(self.raw_path, item))
            if vector is None:
                continue

            # Adding vector number dimension
            for key, value in vector.data.items():
                if key not in self.data:
                    self.data[key] = []
                self.data[key].append(value)

            if verbose:
                bar(i, prefix=f'{sysinfo.get()}')

        for key, value in self.data.items():
            self.data[key] = np.array(value)

        if verbose:
            bar(i+1)
            logs.info(f"Loaded {self.name} dataset from raw data ✅")

    # Get vectors -------------------------------------------------------------

    def __getitem__(self, i):
        """Get vector(s) of the dataset"""

        res = copy(self)

        for key, value in res.data.items():
            res.data[key] = value[i]

        return res
    
    # Get number of vectors in the dataset ------------------------------------

    def __len__(self):
        """Get the number of vector in the dataset"""
        return list(self.data.values())[0].shape[0]

    @property
    def size(self):
        """Get the number of vector in the dataset"""
        return len(self)

    @property
    def shape(self):
        return len(self), len(self.data)
    
    # Get field labels --------------------------------------------------------
    
    @property
    def labels(self):
        """Get the x labels"""
        return list(self.data.keys())
    
    # Get field shapes --------------------------------------------------------
    
    @property
    def shapes(self):
        """Get the shape of each field"""
        return [i.shape[1:] for i in self.data.values()]

    # Get input or output dict ------------------------------------------------

    @property
    def x(self):
        """Get the x dict"""

        if self.xlabels is None:
            raise ValueError("xlabels attribute must be set before accessing x dict")

        x = self.filter(self.xlabels)
        return x

    @property
    def y(self):
        """Get the y dict"""

        if self.ylabels is None:
            raise ValueError("ylabels attribute must be set before accessing y dict")

        y = self.filter(self.ylabels)
        return y

    # Get sub sets ------------------------------------------------------------

    @property
    def train(self):
        """Get the train subset"""
        return self[np.arange(int(len(self)*(1-self.val_frac-self.test_frac)))]
    
    @property
    def val(self):
        """Get the val subset"""
        return self[np.arange(int(len(self)*(1-self.val_frac-self.test_frac)),int(len(self)*(1-self.test_frac)))]

    @property
    def test(self):
        """Get the test subset"""
        return self[np.arange(int(len(self)*(1-self.test_frac)),len(self))]
    
    # Get string representation -----------------------------------------------

    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        """Get printable-ready data summary"""

        max_label_length = max([len(i) for i in self.labels])
        
        res = f"{self.name} dataset, containing {len(self)} vectors."
        if not self.is_processed():
            res += " (Not processed, no statistics available)"
        if self.train is not None and self.val is not None and self.test is not None:
            res += f"\nSubsets: Train: {len(self.train)} vectors, Val: {len(self.val)} vectors, Test: {len(self.test)} vectors."
        
        res += "\nFields:"

        for key in self.labels:
            res += f"\n   - {key + ' ' * (max_label_length - len(key))}"

            m = float(self.means[key])
            s = float(self.stds[key])
            mi = float(self.mins[key])
            ma = float(self.maxs[key])

            if self.is_processed():
                res += f"   Mean: {(' ' if m >= 0 else '')}{m:.2e}"
                res += f"   Std: {(' ' if s >= 0 else '')}{s:.2e}"
                res += f"   Min: {(' ' if mi >= 0 else '')}{mi:.2e}"
                res += f"   Max: {(' ' if ma >= 0 else '')}{ma:.2e}"
            res += f"   Shape: {self.shapes[self.labels.index(key)]}"

        return res

    def print_few_vectors(self, count=5) -> str:
        """Print a resume of few vectors in the dataset"""

        max_label_length = max([len(i) for i in self.labels])
        
        res = f"Here is {count} vectors from the {self.name} dataset:"
     
        for i in range(count):

            r = np.random.randint(0, len(self))
            vector = self[r]

            res += f"\nVector {r}"

            for key in self.labels:
                res += f"\n   - {key + ' ' * (max_label_length - len(key))}"

                mean = self.means[key]
                std = self.stds[key]
                if std == 0:
                    std = 1

                m = np.mean(self.denormalize(key, vector.data[key]))
                s = np.std(self.denormalize(key, vector.data[key]))
                mi = np.min(self.denormalize(key, vector.data[key]))
                ma = np.max(self.denormalize(key, vector.data[key]))

                if vector.data[key].shape != (1,):
                    res += f"   Mean: {(' ' if m >= 0 else '')}{m:.2e}"
                    res += f"   Std: {(' ' if s >= 0 else '')}{s:.2e}"
                    res += f"   Min: {(' ' if mi >= 0 else '')}{mi:.2e}"
                    res += f"   Max: {(' ' if ma >= 0 else '')}{ma:.2e}"
                    res += f"   Shape: {vector.data[key].shape}"
                else:
                    res += f"   Value: {self.denormalize(key, vector.data[key])[0]:.2e}"

        print(res)
        return res

    # Process the data (normalize and split) ----------------------------------

    def process(self, verbose:bool=True) -> "Dataset":
        """Process the dataset"""

        self._normalize(verbose=verbose)
        self._shuffle(uniform_tests_indices=True,verbose=verbose)

        self._processed = True

        return self    
    
    def is_processed(self):
        return self._processed
    
    # Normalization -----------------------------------------------------------

    def _normalize(self, verbose:bool=True):
        """Normalize the dataset"""
        if verbose:
            logs.info(f"Normalizing {self.name}'s Dataset...")

        bar = progress.Bar(len(self.data))

        for key, value in self.data.items():
            self.means[key] = np.mean(value)
            self.stds[key] = np.std(value)
            self.mins[key] = np.min(value)
            self.maxs[key] = np.max(value)
            self.data[key] = self.normalize(key, value)
            bar(bar.previous_progress[-1]+1, prefix=sysinfo.get())

        bar(len(self.data))

        if verbose:
            logs.info(f"{self.name} dataset normalized ✅")

        return self
    
    def normalize(self, key, value):
        return (value + self.mins[key]) / (self.maxs[key] - self.mins[key])
    # return (value - mean) / std

    def denormalize(self, key, value):
        return value * (self.maxs[key] - self.mins[key]) - self.mins[key]
    # return value * std + mean
    
    # Shuffle the dataset -----------------------------------------------------

    def _shuffle(self, uniform_tests_indices=False, verbose:bool=True):
        """Shuffle the dataset"""

        if verbose:
            logs.info(f"Shuffling {self.name}'s Dataset...")

        # Get random indices
        idx = np.random.permutation(len(self))

        if uniform_tests_indices:
            # Follow issue #7

            indices = [0] # Extract first element of the dataset 

            N = len(self)
            N2 = N - 2
            T = int(self.test_frac * N)
            T2 = T - 2

            for i in range(T2):
                indices.append(int((i+1) * N2 / (T2 + 1))+1) # Extract all middle elements
            
            indices.append(N-1) # Extract last element of the dataset 
            indices=np.array(indices)

            # Extract the indices of the test set
            for i in indices:
                idx = np.delete(idx, np.where(idx == i))

            # Put  the indices of the test set at the end (where the test set is taken from)
            idx = np.concatenate((idx , indices), axis=0)

        # Shuffle the dataset using the randomized indices
        self = self[idx]

        if verbose:
            logs.info(f"{self.name} dataset shuffled ✅")

        return self
    
    # Filter the dataset ------------------------------------------------------

    def filter(self, labels:list[str]=None, verbose=False):
        """Filter the dataset"""

        if isinstance(labels, str):
            labels = [labels]

        if verbose:
            logs.info(f"Filtering {self.name} dataset...")

        filtered = copy(self)
        filtered.name = f"{self.name} filtered"
        for label in self.labels:
            if label not in labels:
                del filtered.data[label]
                del filtered.means[label]
                del filtered.stds[label]
                del filtered.mins[label]
                del filtered.maxs[label]

        if verbose:
            logs.info(f"{self.name} dataset filtered ✅\n{filtered}")

        return filtered


    

    
"""
██╗   ██╗███╗   ██╗██╗████████╗    ████████╗███████╗███████╗████████╗███████╗
██║   ██║████╗  ██║██║╚══██╔══╝    ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔════╝
██║   ██║██╔██╗ ██║██║   ██║          ██║   █████╗  ███████╗   ██║   ███████╗
██║   ██║██║╚██╗██║██║   ██║          ██║   ██╔══╝  ╚════██║   ██║   ╚════██║
╚██████╔╝██║ ╚████║██║   ██║          ██║   ███████╗███████║   ██║   ███████║
 ╚═════╝ ╚═╝  ╚═══╝╚═╝   ╚═╝          ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚══════╝                                                                  
"""

import unittest

class TestStringMethods(unittest.TestCase):

    test_x = {
        "x1": 0,
        "x2": np.array([1, 2, 3, 4, 5]),
        "x3": np.array([[[-1,-2,-3],[-4,-5,-6],[-7,-8,-9]],[[1,2,3],[4,5,6],[7,8,9]]]),
    }

    test_y = {
        "y1": 0,
        "y2": np.array([1, 2, 3, 4, 5]),
        "y3": np.array([[[-1,-2,-3],[-4,-5,-6],[-7,-8,-9]],[[1,2,3],[4,5,6],[7,8,9]]]),
    }

    # TODO

if __name__ == '__main__':
    unittest.main()