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
import type_enforced
from typing import Union
import os
from LRFutils import progress
from . import sysinfo

# @type_enforced.Enforcer
class Dataset():

    def __init__(self, name:str='Unammed', loader=None, raw_path:Union[str,None]=None, archive_path:Union[str,None]=None, x:Union[dict[np.ndarray],None]=None, y:Union[dict[np.ndarray],None]=None, autosave:bool=True, silent:bool=False) -> "Dataset":
        """Create a user-friendly dataset"""

        if loader is not None and loader is not callable:
            raise TypeError('"loader" must be a callable object')

        self.name = name
        self.archive_path = archive_path

        if loader is not None and raw_path is not None:
            self.loader = loader
            self.raw_path = raw_path
            self.x, self.y = {}, {}
            self.load(verbose = not silent)

        elif x is not None and y is not None:
            self.x = x
            self.y = y
            Dataset._checkstruct(x, y)

        else:
            raise ValueError('Either "loader" and "raw_path" or "x" and "y" must be provided.')
        
        self.process()
        
        if autosave:
            self.save()

    def load(self, verbose=True):
        """Load data from indicated data pathes"""

        if os.path.isfile(self.archive_path):
            if verbose:
                print(f'Found dataset archive, loading it: {self.archive_path}...')
            self.load_archive()
        
        if verbose:
            print(f'No dataset archive found, loading raw data from: {self.raw_path}...')
        self.load_raw()

        return self.x, self.y
    
    def load_raw(self, verbose=True):
        """Load data from raw data path"""

        bar = progress.Bar(len(os.listdir(self.raw_path)))
        bar(0)

        for i, item in os.listdir(self.raw_path):
            x, y = self.loader(item)
            self += (x, y)
            bar(i+1, prefix=f'{sysinfo.get()}')

        return x, y


    def __getitem__(self, key):
        """Get elements of the dataset that match the key (can be either vector number of element label)"""

        # Raise exception if the key type is not supported
        if not (isinstance(key, str)
                or isinstance(key, int)
                or isinstance(key, slice)
                or isinstance(key, list[int])
                or isinstance(key, np.ndarray[int])
            ):
            raise TypeError('"key" must be a string, an integer, a slice, a list/numpy array of integers')

        # Return the dict elements that match the key
        if isinstance(key, str):
            if key in self.x and not key in self.y:
                return self.x[key]
            if key in self.y and not key in self.x:
                return self.y[key]
            if key in self.x and key in self.y:
                return self.x[key], self.y[key]
            else:
                return None
        
        # Return the vector(s) that match the key
        return self.x[key], self.y[key]
    
    @staticmethod
    def _checkdict(x, y):
        """Check if x and y are dict"""

        if not isinstance(x, dict[np.ndarray]) or not isinstance(y, dict[np.ndarray]):
            raise TypeError('x and y must be dictiononary of numpy ndarrays. Dict elements are inputs/outputs data types and first dimension of ndarrays correspond to the vector number.')

    @staticmethod
    def _checkdim(x, y):
        """Check length of x and y vectors"""

        sample_value = list(x.values())[0]
        if any([i.shape != sample_value.shape for i in  x.values()]) or any([i != x[0] for i in  y.values()]):
            raise ValueError('\nAll x and y vectors of a dataset must have the same shape. The shape should be (nb_vectors, *data_shape, ).\n - "nb_vectors" must be the same on all the inputs and outputs dict keys.\n - "data_shape" must be consistent for a given dict key.\n')

    @staticmethod
    def _checkstruct(x, y):
        """Overhaul check if the dataset is correctly structured"""

        Dataset._checkdict(x, y)
        Dataset._checkdim(x, y)

    def __len__(self):
        """Get the number of vector in the dataset"""
        return list(self.x.values())[0].shape[0]

    @property
    def size(self):
        """Get the number of vector in the dataset"""
        return len(self)
    
    @property
    def xlabels(self):
        """Get the x labels"""
        return list(self.x.keys())
    
    @property
    def ylabels(self):
        """Get the y labels"""
        return list(self.y.keys())
    
    @property
    def xshapes(self):
        """Get the x shapes"""
        return [i.shape[1:] for i in self.x.values()]

    @property
    def yshapes(self):
        """Get the y shapes"""
        return [i.shape[1:] for i in self.y.values()]
    
    def __repr__(self) -> str:
        res = f"{self.name} dataset, containing {len(self)} vectors."
        res += "\nInput(s):"
        for label, shape in zip(self.xlabels, self.xshapes):
            res += f"\n - {label} {shape}"
        res += f"\nOutput(s):"
        for label, shape in zip(self.ylabels, self.yshapes):
            res += f"\n - {label} {shape}"

    def __str__(self) -> str:
        return self.__repr__()

    def add(self, x:np.ndarray, y:np.ndarray):
        """Add data to the dataset"""

        Dataset._checkstruct(x, y)
        self.x.update(x)
        self.y.update(y)
        Dataset._checkstruct(self.x, self.y)

    def __add__(self, dataset:Union["Dataset", tuple[dict[np.ndarray], dict[np.ndarray]]]):
        """Add data to the dataset"""

        if isinstance(dataset, Dataset):
            self.merge(dataset)
        else:
            self.add(*dataset)

    def merge(self, dataset:"Dataset"):
        """Merge two datasets"""

        self.add(dataset.x, dataset.y)

    def save(self, path:str):
        """Save the dataset to a numpy compressed file"""

        np.savez_compressed(path, x=self.x, y=self.y)

    @staticmethod
    def load(path:str):
        """Load a dataset from a numpy compressed file"""

        data = np.load(path)
        return Dataset(x=data['x'], y=data['y'])