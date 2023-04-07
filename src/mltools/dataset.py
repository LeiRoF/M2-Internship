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

class Dataset():

    def __init__(self,
                 name:str='Unammed',
                 loader:"function"=None,
                 raw_path:Union[str,None]=None,
                 archive_path:Union[str,None]=None, 
                 x:Union[dict[np.ndarray],None]=None,
                 y:Union[dict[np.ndarray],None]=None, 
                 val_frac:float=0.2,
                 test_frac:float=0.1,
                 autosave:bool=True,
                 process=True,
                 verbose:bool=False,
                 parent:Union["Dataset",None]=None,
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
        self.parent:Dataset = parent

        if 0 > val_frac + test_frac >= 1:
            raise ValueError("val_frac, test_frac and the sum of both must be between 0 and 1.")

        self.val_frac:float = val_frac
        self.test_frac:float = test_frac

        self.x:dict[np.ndarray] = {}
        self.xmeans:dict[float] = {}
        self.xstds:dict[float] = {}
        self.xmins:dict[float] = {}
        self.xmaxs:dict[float] = {}

        self.y:dict[np.ndarray] = {}
        self.ymeans:dict[float] = {}
        self.ystds:dict[float] = {}
        self.ymins:dict[float] = {}
        self.ymaxs:dict[float] = {}
        
        self._processed:bool = False if not parent else self.parent.is_processed()

        self.train:Dataset = None
        self.val:Dataset = None
        self.test:Dataset = None

        # Data loading
        if x or y:
            if not (x and y):
                raise ValueError("To create a dataset with already loaded data, both x and y must be given.")
            if loader or raw_path or archive_path:
                raise ValueError("Can't load data from multiple source at once. You must create a dataset using either x and y (already loaded data), either data pathes.")

            self.x = x
            self.y = y
        
        else:

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

        # Process and save files
        if self.parent is None:

            if process:
                self.process(verbose=verbose)
            
            if autosave and archive_path is not None:
                logs.warn("Save archive is not implemented yet.")
                # self.save()

        # If there is a parent dataset, inherit its attributes
        else:
            if self.parent.is_processed():
                for key in self.x.keys():
                    self.xmeans[key] = self.parent.xmeans[key]
                    self.xstds[key] = self.parent.xstds[key]
                    self.xmins[key] = self.parent.xmins[key]
                    self.xmaxs[key] = self.parent.xmaxs[key]
                for key, value in self.y.items():
                    self.ymeans[key] = self.parent.ymeans[key]
                    self.ystds[key] = self.parent.ystds[key]
                    self.ymins[key] = self.parent.ymins[key]
                    self.ymaxs[key] = self.parent.ymaxs[key]

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

        self.x = {}
        self.y = {}

        if verbose:
            logs.info(f"Loading {self.name}'s raw data from {self.raw_path}...")
            bar = progress.Bar(len(files))

        for i, item in enumerate(files):

            try:
                x, y = self.loader(os.path.join(self.raw_path, item))
            except:
                continue

            # Adding vector number dimension
            for key, value in x.items():
                if key not in self.x:
                    self.x[key] = []
                self.x[key].append(value)

            for key, value in y.items():
                if key not in self.y:
                    self.y[key] =  []
                self.y[key].append(value)

            if verbose:
                bar(i, prefix=f'{sysinfo.get()}')

        for key, value in self.x.items():
            self.x[key] = np.array(value)
        
        for key, value in self.y.items():
            self.y[key] = np.array(value)        

        if verbose:
            bar(i+1)
            logs.info(f"Loaded {self.name} dataset from raw data ✅")

    # Get vectors -------------------------------------------------------------

    def __getitem__(self, key):
        """Get elements of the dataset that match the key (can be either vector number of element label)"""

        # Raise exception if the key type is not supported
        if not (isinstance(key, str)
                or isinstance(key, int)
                or isinstance(key, slice)
                or isinstance(key, list)
                or isinstance(key, np.ndarray)
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
        
        res_x = {}
        res_y = {}

        for field, data in self.x.items():
            res_x[field] = data[key]
        for field, data in self.y.items():
            res_y[field] = data[key]
        
        return Dataset(name=f"{self.name} subset", x=res_x, y=res_y, parent=self)
    
    # Check if dataset element are dictionaries -------------------------------

    @staticmethod
    def _checkdict(**kwargs):
        """Check if x and y are dict"""

        for set, x in kwargs.items():
            if not isinstance(x, dict):
                raise TypeError(f"{set} set is not a dictionary.")
            
    # Check if the number of vectors is consistent between fields -------------

    @staticmethod
    def _checksize(**kwargs):
        """Check length of vectors"""

        for set, x in kwargs.items():
            sample_key = list(x.keys())[0]
            if any([i.shape[0] != x[sample_key].shape[0] for i in  x.values()]):
                raise ValueError(f'\nSet {set} contain inconsistent number of vectors. Found {[i.shape[0] for i in x.values()]}. Maybe the first dimension of your data are not the vector number?\nCurrent element shapes: {[i.shape for i in x.values()]}')

    # Check if the dataset is correctly structured ----------------------------

    @staticmethod
    def _checkstruct(x, y):
        """Overhaul check if the dataset is correctly structured"""

        Dataset._checkdict(x=x, y=y)
        Dataset._checksize(x=x, y=y)

    # Get number of vectors in the dataset ------------------------------------

    def __len__(self):
        """Get the number of vector in the dataset"""
        return list(self.x.values())[0].shape[0]

    @property
    def size(self):
        """Get the number of vector in the dataset"""
        return len(self)
    
    # Get field labels --------------------------------------------------------
    
    @property
    def xlabels(self):
        """Get the x labels"""
        return list(self.x.keys())
    
    @property
    def ylabels(self):
        """Get the y labels"""
        return list(self.y.keys())
    
    # Get field shapes --------------------------------------------------------
    
    @property
    def xshapes(self):
        """Get the x shapes"""
        return [i.shape[1:] for i in self.x.values()]

    @property
    def yshapes(self):
        """Get the y shapes"""
        return [i.shape[1:] for i in self.y.values()]
    
    # Get string representation -----------------------------------------------

    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        """Get printable-ready data summary"""

        max_label_length = max(max([len(i) for i in self.xlabels]), max([len(i) for i in self.ylabels]))
        
        res = f"{self.name} dataset, containing {len(self)} vectors."
        if not self.is_processed():
            res += " (Not processed, no statistics available)"
        if self.train is not None and self.val is not None and self.test is not None:
            res += f"\nSubsets: Train: {len(self.train)} vectors, Val: {len(self.val)} vectors, Test: {len(self.test)} vectors."
        
        res += "\nInputs:"

        for key in self.xlabels:
            res += f"\n - {key + ' ' * (max_label_length - len(key))}"

            if self.is_processed():
                res += f"   Mean: {(' ' if float(self.xmeans[key]) >= 0 else '')}{float(self.xmeans[key]):.2e}"
                res += f"   Std: {(' ' if float(self.xstds[key]) >= 0 else '')}{float(self.xstds[key]):.2e}"
                res += f"   Min: {(' ' if float(self.xmins[key]) >= 0 else '')}{float(self.xmins[key]):.2e}"
                res += f"   Max: {(' ' if float(self.xmaxs[key]) >= 0 else '')}{float(self.xmaxs[key]):.2e}"
            res += f"   Shape: {self.xshapes[self.xlabels.index(key)]}"

        res += "\nOutputs:"
        for key in self.ylabels:
            res += f"\n - {key + ' ' * (max_label_length - len(key))}"

            if self.is_processed():
                res += f"   Mean: {(' ' if float(self.ymeans[key]) >= 0 else '')}{float(self.ymeans[key]):.2e}"
                res += f"   Std: {(' ' if float(self.ystds[key]) >= 0 else '')}{float(self.ystds[key]):.2e}"
                res += f"   Min: {(' ' if float(self.ymins[key]) >= 0 else '')}{float(self.ymins[key]):.2e}"
                res += f"   Max: {(' ' if float(self.ymaxs[key]) >= 0 else '')}{float(self.ymaxs[key]):.2e}"
            res += f"   Shape: {self.yshapes[self.ylabels.index(key)]}"

        return res
    
    # Add vector to a dataset -------------------------------------------------

    # def add(self, x:np.ndarray, y:np.ndarray):
    #     """Add data to the dataset"""

    #     for key, value in x.items():
    #         if key in self.x:
    #             self.x[key] = np.concatenate((self.x[key], value), axis=0)
    #         else:
    #             self.x[key] = value
    #     for key, value in y.items():
    #         if key in self.y:
    #             self.y[key] = np.concatenate((self.y[key], value), axis=0)
    #         else:
    #             self.y[key] = value

    #     Dataset._checkstruct(self.x, self.y)

    #     return self

    # def __add__(self, dataset:Union["Dataset", tuple[dict[np.ndarray], dict[np.ndarray]]]):
    #     """Add data to the dataset"""

    #     if isinstance(dataset, Dataset):
    #         self.merge(dataset)
    #     else:
    #         self.add(*dataset)
        
    #     return self

    # def merge(self, dataset:"Dataset"):
    #     """Merge two datasets"""

    #     self.add(dataset.x, dataset.y)

    #     return self    

    # Process the data (normalize and split) ----------------------------------

    def process(self, verbose:bool=True) -> "Dataset":
        """Process the dataset"""

        self.normalize(verbose=verbose)
        self.shuffle(uniform_tests_indices=True,verbose=verbose)
        self.split(verbose=verbose)

        self._processed = True

        return self    
    
    def is_processed(self):
        return self._processed
    
    # Normalization -----------------------------------------------------------

    def normalize(self, verbose:bool=True):
        """Normalize the dataset"""
        if verbose:
            logs.info(f"Normalizing {self.name}'s Dataset...")

        self.xmeans = {}
        self.xstds = {}
        self.xmins = {}
        self.xmaxs = {}
        self.ymeans = {}
        self.ystds = {}
        self.ymins = {}
        self.ymaxs = {}

        bar = progress.Bar(len(self.x) + len(self.y))

        for key, value in self.x.items():
            self.xmeans[key] = np.mean(value)
            self.xstds[key] = np.std(value)
            self.xmins[key] = np.min(value)
            self.xmaxs[key] = np.max(value)
            self.x[key] = (value + self.xmins[key]) / (self.xmaxs[key] - self.xmins[key])
            # self.x[key] = (value - self.xmeans[key]) / self.xstds[key]
            bar(bar.previous_progress[-1]+1, prefix=sysinfo.get())

        for key, value in self.y.items():
            self.ymeans[key] = np.mean(value)
            self.ystds[key] = np.std(value)
            self.ymins[key] = np.min(value)
            self.ymaxs[key] = np.max(value)
            self.y[key] = (value + self.ymins[key]) / (self.ymaxs[key] - self.ymins[key])
            # self.y[key] = (value - self.ymeans[key]) / self.ystds[key]
            bar(bar.previous_progress[-1]+1, prefix=sysinfo.get())

        bar(len(self.x) + len(self.y))

        if verbose:
            logs.info(f"{self.name} dataset normalized ✅")

        return self
    
    # Shuffle the dataset -----------------------------------------------------

    def shuffle(self, uniform_tests_indices=False, verbose:bool=True):
        """Shuffle the dataset"""

        if verbose:
            logs.info(f"Shuffling {self.name}'s Dataset...")

        # Get random indices
        idx = np.random.permutation(len(self))
        print(idx.shape)
        print(idx)

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
    
    # Split the dataset -------------------------------------------------------

    def split(self, verbose:bool=True):
        """Split the dataset into train, validation and test sets"""

        if verbose:
            logs.info(f"Splitting {self.name}'s Dataset...")

        N = len(self)
        self.train = self[:int(N*(1-self.val_frac-self.test_frac))]
        self.train.name = f"{self.name} train"
        self.val = self[int(N*(1-self.val_frac-self.test_frac)):int(N*(1-self.test_frac))]
        self.val.name = f"{self.name} val"
        self.test = self[int(N*(1-self.test_frac)):]
        self.test.name = f"{self.name} test"

        if verbose:
            logs.info(f"{self.name} dataset splitted ✅\n - Train set: {len(self.train)} vectors\n - Validation set: {len(self.val)} vectors\n - Test set: {len(self.test)} vectors")

        return self.train, self.val, self.test
    
    # Filter the dataset ------------------------------------------------------

    def filter(self, xlabels:list[str]=None, ylabels:list[str]=None, verbose=False):
        """Filter the dataset"""

        if verbose:
            logs.info(f"Filtering {self.name} dataset...")
    
        if xlabels is not None:
            xlabels = [label.lower().replace(" ", "_") for label in xlabels]

        filtered_x = {}
        for label in self.xlabels:
            if label.lower().replace(" ", "_") in xlabels:
                filtered_x[label] = self[label]

        if ylabels is not None:
            ylabels = [label.lower().replace(" ", "_") for label in ylabels]

        filtered_y = {}
        for label in self.ylabels:
            if label.lower().replace(" ", "_") in ylabels:
                filtered_y[label] = self[label]

        dataset = Dataset(name=f"{self.name} filtered", x=filtered_x, y=filtered_y, parent=self)
        if self.train is not None:
            dataset.train = self.train.filter(xlabels=xlabels, ylabels=ylabels, verbose=False)
        if self.val is not None:
            dataset.val = self.val.filter(xlabels=xlabels, ylabels=ylabels, verbose=False)
        if self.test is not None:
            dataset.test = self.test.filter(xlabels=xlabels, ylabels=ylabels, verbose=False)

        if verbose:
            logs.info(f"{self.name} dataset filtered ✅\n{dataset}")

        return dataset


    

    
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