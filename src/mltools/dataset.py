import numpy as np
from typing import Union, Callable
import os
from LRFutils import progress, logs
import sysinfo
from vector import Vector
import matplotlib.pyplot as plt
from copy import deepcopy as copy
from multiprocessing import Pool

from typing import Union, Callable, List, Dict, Tuple, Any

#==========================================================================
# DATASET
#==========================================================================

class Dataset(Dict):

    def __init__(self, vectors:List[Vector]=None, name:str="Unnamed"):
        """
        Create a user-friendly dataset.
        - loader & raw path allow to load raw data
        - If archive_path is given and the file exist, it will ignore the raw path.
        """
        super().__init__()

        self._name = name

        if vectors is not None:
            if isinstance(vectors, Vector):
                vectors = [vectors]
            for label in vectors[0].keys():
                self[label] = np.array([vector[label] for vector in vectors])
        
        self["index"] = np.arange(len(vectors))

        self._normalized = False
        self._splitted = False

        self._means = {}
        self._stds = {}
        self._maxs = {}
        self._mins = {}

        self._train = None
        self._test = None
        self._validation = None

    # Name -------------------------------------------------------------------

    @property
    def name(self):
        """Get the name of the dataset

        Returns:
            str: the name of the dataset
        """
        return self._name
    
    @name.setter
    def name(self, value:str):
        """Get the name of the dataset

        Args:
            value (str): the new name of the dataset

        Raises:
            TypeError: if value is not a string
        """
        if not isinstance(value, str):
            raise TypeError(f"name must be a string, not {type(value)}")
        self.name = value

    # Get vectors -------------------------------------------------------------

    def get(self, i):
        """Select item(s) from dataset
        
        Args:
            i (str|int|slice): the index to select

        Raises:
            TypeError: if i is not a string, an int or a slice

        Returns:
            numpy.ndarray | Vector | Dataset: the selected column | vector | subset of the dataset
        """

        # Dictionnary behavior
        if isinstance(i,str):
            return super().__getitem__(i)

        # Simple vector selection
        elif isinstance(i, int):
            res = Vector()
            for key, value in self.items():
                if key == "index":
                    continue
                res[key] = value[i]
            return res

        # Numpy behavior
        else:
            try:
                res = copy(self)
                for key, value in res.items():
                    res[key] = value[i]
                return res
            except:
                TypeError(f"Invalid index type: {type(i)}. It must be either string, int or anything that can be used to index a 1D numpy array")
    
    def __getitem__(self, i):
        """Select item(s) from dataset
        
        Args:
            i (str|int|slice): the index to select

        Raises:
            TypeError: if i is not a string, an int or a slice

        Returns:
            numpy.ndarray | Vector | Dataset: the selected column | vector | subset of the dataset

        Alias:
            get
        """
        return self.get(i)

    # Get index ---------------------------------------------------------------

    def index(self, vector):
        """Get the index of a vector in the dataset

        Args:
            Vector (Vector): the vector to find

        Raises:
            TypeError: if Vector is not a Vector

        Returns:
            list[int]: the lsit of index where the vector is found
        """

        if not isinstance(vector, Vector):
            raise TypeError(f"vector must be a Vector, not {type(Vector)}")
        
        res = []

        for i in range(len(self)):

            match = 0
            for label in self.labels:
                if label == "index":
                    continue
                if label not in vector.keys():
                    raise TypeError(f"vector must have the same labels as the dataset, found vector label: {vector.keys()} and dataset labels: {self.labels}")
          
                if self[label][i] == vector[label]:
                    match += 1

            if match == len(self.labels) - 2:
                res.append(i)
            
        return res

    # Add vectors -------------------------------------------------------------

    def append(self, vector:Vector):
        """Append a vector to the dataset
        
        Args:
            vector (Vector): the vector to append

        Raises:
            TypeError: if vector is not a Vector 

        Returns:
            Dataset: the dataset   
        """
        if not isinstance(vector, Vector):
            raise TypeError(f"vector must be a Vector, not {type(vector)}")

        if self.keys() != vector.keys():
            raise TypeError(f"vector must have the same keys as the dataset, found vector label: {vector.keys()} and dataset labels: {self.keys()}")
        
        was_normalized = self.is_normalized()
        self.denormalize()

        for key, value in vector.items():
            self[key] = np.append(self[key], value)
        self["index"] = np.append(self["index"], len(self))

        if was_normalized:
            self.normalize()

        return self

    def add(self, vectors:Union[Vector, List[Vector]]):
        """Append a vector to the dataset
        
        Args:
            vector (Vector): the vector to append

        Raises:
            TypeError: if vector is not a Vector

        Returns:
            Dataset: the dataset
        """
        if isinstance(vectors, Vector):
            self.append(vectors)
        
        elif isinstance(vectors, list):
            
            was_normalized = self.is_normalized()
            self.denormalize()

            for vector in vectors:
                if not isinstance(vector, Vector):
                    raise TypeError(f"all vectors must be a Vector, found {type(vector)}")
                self.append(vector)

            if was_normalized:
                self.normalize()
        
        else:
            raise TypeError(f"vectors must be a Vector or a list of Vector, not {type(vectors)}")
        
        return self

    # Remove vectors ----------------------------------------------------------

    def pop(self, i:int):
        """Remove a vector from the dataset and return it

        Args:
            i (int): the index of the vector to remove
        
        Returns:
            Vector: the removed vector

        Raises:
            TypeError: if i is not an int
        """
        if not isinstance(i, int):
            raise TypeError(f"i must be an int, not {type(i)}")

        was_normalized = self.is_normalized()
        self.denormalize()

        popped = Vector()
        for key, value in self.items():
            popped[key] = value[i]
            self[key] = np.delete(value, i)

        if was_normalized:
            self.normalize()

        return popped

    def remove(self, i:Union[int, Vector, List[int], List[Vector]]):
    
        if not isinstance(i, int) or not isinstance(i, Vector):
            i = [i]
        
        for j in i:
            if not isinstance(j, int) or not isinstance(j, Vector):
                raise TypeError(f"i must be an int, a Vector, a list of int or a list of Vector, not {type(i)}")

            # TODO
            

    # Get info about dataset shape --------------------------------------------

    @property
    def lines(self):
        """Get the number of vector in the dataset
        
        Returns:
            int: the number of vector in the dataset
        """
        return list(self.values())[0].shape[0]
    
    def __len__(self):
        """Get the number of vector in the dataset
        
        Returns:
            int: the number of vector in the dataset

        Alias:
            lines
        """
        return self.lines
    
    @property
    def columns(self):
        """Get the number of vector in the dataset
        
        Returns:
            int: the number of vector in the dataset
        """
        return super().__len__()

    @property
    def shape(self):
        """Get the number of vector in the dataset

        Returns:
            int: the number of vector in the dataset
        """
        return self.lines * self.columns
    
    @property
    def size(self):
        """Get the number of vector in the dataset
        
        Returns:
            int: the number of vector in the dataset
        """
        return self.lines * self.columns
    
    # Get field labels --------------------------------------------------------
    
    @property
    def labels(self):
        """Get the labels
        
        Returns:
            list[str]: the labels
        """
        labels = list(self.keys())
        labels.remove("index")
        return labels
    
    # Get field shapes --------------------------------------------------------
    
    @property
    def shapes(self):
        """Get the shape of each field"""
        shapes = []
        for label, value in self.items():
            if label == "index":
                continue
            shapes.append(value.shape[1:])
        return shapes

    # Get normalization infos -------------------------------------------------

    def is_normalized(self):
        """Check if the dataset is normalized

        Returns:
            bool: True if the dataset is normalized, False otherwise
        """
        return self._normalized

    def mean(self, field:str=None):
        """Get the mean of each field
        
        Args:
            field (str, optional): the field to get the mean from. Defaults to None.

        Raises:
            TypeError: if field is not a string or None

        Returns:
            float | dict[str, float]: the mean of the specified field | the mean of each field as a dict
        """
        if field is not None:
            if not isinstance(field, str):
                raise TypeError(f"field must be a string, not {type(field)}")
            return self._means[field]
        return self._means
    
    def std(self, field:str=None):
        """Get the standard deviation of each field
        
        Args:
            field (str, optional): the field to get the standard deviation from. Defaults to None.

        Raises:
            TypeError: if field is not a string or None

        Returns:
            float | dict[str, float]: the standard deviation of the specified field | the standard deviation of each field as a dict
        """
        if field is not None:
            if not isinstance(field, str):
                raise TypeError(f"field must be a string, not {type(field)}")
            return self._stds[field]
        return self._stds
    
    def max(self, field:str=None):
        """Get the max of each field
        
        Args:
            field (str, optional): the field to get the max from. Defaults to None.

        Raises:
            TypeError: if field is not a string or None

        Returns:
            float | dict[str, float]: the max of the specified field | the max of each field as a dict
        """
        if field is not None:
            if not isinstance(field, str):
                raise TypeError(f"field must be a string, not {type(field)}")
            return self._maxs[field]
        return self._maxs

    def min(self, field:str=None):
        """Get the min of each field
        
        Args:
            field (str, optional): the field to get the min from. Defaults to None.

        Raises:
            TypeError: if field is not a string or None

        Returns:
            float | dict[str, float]: the min of the specified field | the min of each field as a dict
        """
        if field is not None:
            if not isinstance(field, str):
                raise TypeError(f"field must be a string, not {type(field)}")
            return self._mins[field]
        return self._mins

    # Dataset normalization ---------------------------------------------------

    def normalize(self):
        """Normalize the dataset

        Returns:
            Dataset: the normalized dataset
        """

        if self.is_normalized():
            return
        
        for key, value in self.items():
            if key == "index":
                continue

            std = value.std()
            if std == 0:
                std = 1
            self[key] = (value - value.mean()) / value.std()

            self._means[key] = value.mean()
            self._stds[key] = value.std()
            self._maxs[key] = value.max()
            self._mins[key] = value.min()

        self._normalized = True

        return self

    def denormalize(self, label=None, value=None):
        """Denormalize the dataset, a colummn if the label is specified, or a given value if the label and the value are specified

        Raises:
            ValueError: if the label is "index"
            ValueError: if the label is not specified and the value is specified
            ValueError: if the dataset is not normalized and the value is specified
            TypeError: if the label is not a string or None
        
        Returns:
            Dataset | numpy.ndarray | object: the denormalized dataset | column | value
        """

        if label is None and value is not None:
            raise ValueError("Cannot denormalize a value without a label")
        if not self.is_normalized() and value is not None:
            raise ValueError("not normalized dataset cannot be be used to denormalize a value")
        if label == "index":
            raise ValueError("Cannot denormalize index field")
        if not isinstance(label, str):
            raise TypeError(f"label must be a string, not {type(label)}")

        if label is not None:        
            if value is None:
                value = self[label]
            return value * self._stds[label] + self._means[label]

        if not self.is_normalized():
            return
        
        for key, value in self.items():
            if key == "index":
                continue

            self[key] = value * self._stds[key] + self._means[key]

        self._normalized = False

        return self
    
    # Shuffle -----------------------------------------------------------------

    def shuffle(self, verbose=False):
        """Shuffle the dataset
        
        Returns:
            Dataset: the shuffled dataset
        """

        if verbose:
            logs.info(f"Shuffling {self.name}'s Dataset...")

        # Get random indices
        idx = np.random.permutation(len(self))

        # Shuffle the dataset using the randomized indices
        self = self[idx]

        if verbose:
            logs.info(f"{self.name} dataset shuffled ✅")

        return self
    
    # Get sub sets ------------------------------------------------------------

    def is_splitted(self):
        return self._splitted

    @property
    def train(self):
        """Get the train subset

        Raises:
            ValueError: if the dataset is not splitted
        
        Returns:
            Dataset: the train subset
        """
        if not self.is_splitted():
            raise ValueError("Dataset is not splitted")
        return self._train
    
    @property
    def val(self):
        """Get the val subset

        Raises:
            ValueError: if the dataset is not splitted
        
        Returns:
            Dataset: the val subset
        """
        if not self.is_splitted():
            raise ValueError("Dataset is not splitted")
        return self._val

    @property
    def test(self):
        """Get the test subset

        Raises:
            ValueError: if the dataset is not splitted
        
        Returns:
            Dataset: the test subset
        """
        if not self.is_splitted():
            raise ValueError("Dataset is not splitted")
        return self._test

    # Split dataset -----------------------------------------------------------

    def split(self, val:float, test:float):
        """Split the dataset into train, val and test subsets.

        In addition of being returned, these subsets are also stored in the dataset as attributes: train, val and test.
        
        Args:
            val (float): fraction of the dataset to use for validation
            test (float): fraction of the dataset to use for testing

        Raises:
            TypeError: if val or test are not floats
            ValueError: if val + test is not between 0 and 1

        Returns:
            Dataset: the train subset
            Dataset: the val subset
            Dataset: the test subset
        """

        if not isinstance(val, float) or not isinstance(test, float):
            raise TypeError("val and test must be floats")

        if 0 >= val + test >= 1:
            raise ValueError("val + test must be between 0 and 1")

        index = np.arange(len(self))
        mask = np.ones(len(self), dtype=bool)

        test_step = int(len(self) * test)
        test_mask = mask * ((index % test_step) == 0)
        test_mask[-1] = 1

        self._test = self[test_mask]
        rest = self[~test_mask]

        index = np.arange(len(rest))
        mask = np.ones(len(rest), dtype=bool)

        val_step = int(len(rest) * val / (1-test))
        val_mask = mask * ((index % val_step) == 0)
        val_mask[-1] = 1

        rest.shuffle()
        self._val = rest[val_mask]
        self._train = rest[~val_mask]

        self._splitted = True

        return self._train, self._val, self._test

    # Get string representation -----------------------------------------------

    def __str__(self) -> str:
        """Get data summary as str

        Returns:
            str: the data summary
        """
        return self.__repr__()
    
    def __repr__(self) -> str:
        """Get printable-ready data summary
        
        Returns:
            str: the data summary
        """

        max_label_length = max([len(i) for i in self.labels])
        
        res = f"{self.name} dataset, containing {len(self)} vectors."
        if not self.is_normalized():
            res += " (Not processed, no statistics available)"

        if self.is_splitted():
            res += f"\nSubsets: Train: {len(self.train)} vectors, Val: {len(self.val)} vectors, Test: {len(self.test)} vectors."
        
        res += "\nFields:"

        for label in self.labels:
            if label == "index":
                continue
            res += f"\n   - {label + ' ' * (max_label_length - len(label))}"

            if self.is_normalized():
                m = self.mean(label)
                s = self.std(label)
                mi = self.min(label)
                ma = self.max(label)
            
                res += f"   Mean: {(' ' if m >= 0 else '')}{m:.2e}"
                res += f"   Std: {(' ' if s >= 0 else '')}{s:.2e}"
                res += f"   Min: {(' ' if mi >= 0 else '')}{mi:.2e}"
                res += f"   Max: {(' ' if ma >= 0 else '')}{ma:.2e}"
            res += f"   Shape: {self.shapes[self.labels.index(label)]}"

        return res
    
    # Print few vectors -------------------------------------------------------

    def print_few_vectors(self, count=5) -> str:
        """Print a resume of few vectors in the dataset
        
        Args:
            count (int, optional): number of vectors to print. Defaults to 5.

        Returns:
            str: the resume that is printed
        """

        max_label_length = max([len(i) for i in self.labels])
        
        res = f"Here is {count} vectors from the {self.name} dataset:"
     
        for i in range(count):

            r = np.random.randint(0, len(self))
            vector = self[r]

            res += f"\nVector {r}"

            for label in self.labels:
                res += f"\n   - {label + ' ' * (max_label_length - len(label))}"

                mean = self.mean(label)
                std = self.std(label)
                if std == 0:
                    std = 1

                m = np.mean(self.denormalize(label, vector[label]))
                s = np.std(self.denormalize(label, vector[label]))
                mi = np.min(self.denormalize(label, vector[label]))
                ma = np.max(self.denormalize(label, vector[label]))

                if vector.data[label].shape != (1,):
                    res += f"   Mean: {(' ' if m >= 0 else '')}{m:.2e}"
                    res += f"   Std: {(' ' if s >= 0 else '')}{s:.2e}"
                    res += f"   Min: {(' ' if mi >= 0 else '')}{mi:.2e}"
                    res += f"   Max: {(' ' if ma >= 0 else '')}{ma:.2e}"
                    res += f"   Shape: {vector[label].shape}"
                else:
                    res += f"   Value: {self.denormalize(label, vector[label])[0]:.2e}"

        print(res)
        return res
    
    # Filter the dataset ------------------------------------------------------

    def filter(self, labels:list[str], verbose=False, new_name=None) -> "Dataset":
        """Filter the dataset
        
        Args:
            labels (str, list[str]): the labels to keep.
        
        Raises:
            TypeError: if labels is not a string or a list of strings
            
        Returns:
            Dataset: the filtered dataset
        """

        if verbose:
            logs.info(f"Filtering {self.name} dataset...")

        if isinstance(labels, str):
            labels = [labels]
        
        for label in self.labels:
            if not isinstance(label, str):
                raise TypeError("Labels must be strings")
            
            if label not in labels:
                del self[label]

                if self.is_normalized():
                    del self._mean[label]
                    del self._std[label]
                    del self._min[label]
                    del self._max[label]

        if self.is_splitted():
            self._train = self._train.filter(labels, verbose=verbose)
            self._val = self._val.filter(labels, verbose=verbose)
            self._test = self._test.filter(labels, verbose=verbose)

        if verbose:
            logs.info(f"{self.name} dataset filtered ✅\n{self}")

        if not new_name:
            self.name = f"{self.name} filtered"

        return self

    
"""
██╗   ██╗███╗   ██╗██╗████████╗    ████████╗███████╗███████╗████████╗███████╗
██║   ██║████╗  ██║██║╚══██╔══╝    ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔════╝
██║   ██║██╔██╗ ██║██║   ██║          ██║   █████╗  ███████╗   ██║   ███████╗
██║   ██║██║╚██╗██║██║   ██║          ██║   ██╔══╝  ╚════██║   ██║   ╚════██║
╚██████╔╝██║ ╚████║██║   ██║          ██║   ███████╗███████║   ██║   ███████║
 ╚═════╝ ╚═╝  ╚═══╝╚═╝   ╚═╝          ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚══════╝                                                                  
"""

import unittest

class TestDataset(unittest.TestCase):

    def test_dataset(self):
        v0 = Vector(a=[1,2,3], b=[4,5,6])
        v1 = Vector(a=[7,8,9], b=[10,11,12])
        v2 = Vector(a=[-1,-2,-3], b=[-4,-5,-6])

        d = Dataset([v0, v1, v2], "Test")

        assert d.name == "Test", d.name

        assert d[0] == v0, d[0]
        assert d[1] == v1, d[1]
        assert d[2] == v2, d[2]

        assert d["a"].tolist() == [[1,2,3], [7,8,9], [-1,-2,-3]], d["a"].tolist()
        assert d["b"].tolist() == [[4,5,6], [10,11,12], [-4,-5,-6]], d["b"].tolist()

        assert d.labels == ["a", "b"], d.labels
        assert d.shapes == [(3,), (3,)], d.shapes
        
        d2 = copy(d)

        d2.normalize()

        assert d2.min("a") == -3, d2.min("a")
        assert d2.max("a") == 9, d2.max("a")
        assert d2.mean("a") == 2.6666666666666665, d2.mean("a")
        assert d2.std("a") == 4.189935029992179, d2.std("a")

        d3 = copy(d2)

        assert d3[0] == v0, d3[0]
        assert d3[1] == v1, d3[1]
        assert d3[2] == v2, d3[2]

        

if __name__ == '__main__':
    unittest.main()