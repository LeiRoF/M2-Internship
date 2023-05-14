import numpy as np
from typing import Union, Callable
import os
from LRFutils import progress, logs
from . import sysinfo
from .vector import Vector
import matplotlib.pyplot as plt
from copy import deepcopy as copy
from multiprocessing import Pool
from typing import Union, Callable, List, Dict, Tuple, Any

class Dataset(Dict):

    def __init__(self, vectors:List[Vector]=None, name:str="Unnamed"):
        """Create a dataset from a list of vectors

        Args:
            vectors (List[Vector], optional): the list of vectors to create the dataset from. Defaults to None.
            name (str, optional): the name of the dataset. Defaults to "Unnamed".

        Raises:
            TypeError: if vectors is not a list of Vector

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3})
            ... ])
        """
        super().__init__()

        self._name = name

        self._vectors_uid = np.array([])

        if vectors is not None:
            if isinstance(vectors, Vector):
                vectors = [vectors]
            for vector in vectors:
                self._vectors_uid = np.append(self._vectors_uid, [vector.uid], axis=0)
            for label in vectors[0].keys():
                self[label] = np.array([vector[label] for vector in vectors])

        self._normalized = False

        self._means = {}
        self._stds = {}
        self._maxs = {}
        self._mins = {}

        self._train = None
        self._test = None
        self._val = None

        self._normalization_method = "zscore"

    # Name -------------------------------------------------------------------

    @property
    def name(self):
        """Get the name of the dataset

        Returns:
            str: the name of the dataset

        Examples:
            >>> dataset = Dataset(name="MyData")
            >>> dataset.name
            "MyData"
        """
        return self._name
    
    @name.setter
    def name(self, value:str):
        """Get the name of the dataset

        Args:
            value (str): the new name of the dataset

        Raises:
            TypeError: if value is not a string

        Examples:
            >>> dataset = Dataset(name="MyData")
            >>> dataset.name = "MyNewData"
            >>> dataset.name
            "MyNewData"
        """
        try:
            value = str(value)
        except:
            raise TypeError(f"name must be convertible to a string, which is not the case of {type(value)}")
        self._name = value

    # Get vectors -------------------------------------------------------------

    def get(self, i):
        """Select item(s) from dataset
        
        Args:
            i (str|int|slice): the index to select

        Raises:
            TypeError: if i is not a string, an int or a slice

        Returns:
            numpy.ndarray | Vector | Dataset: the selected column | vector | subset of the dataset

        Alias:
            __getitem__

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.get("x")
            [[1, 2, 3], [-1, -2, -3]]
            >>> dataset.get(0)
            Vector({"x":[1,2,3], "y":2})
            >>> dataset.get(0:2)
            Dataset([
                Vector({"x":[1,2,3], "y":2}),
                Vector({"x":[-1,-2,-3], "y":3}),
            ])
        """

        # Dictionnary behavior
        if isinstance(i,str):
            return super().__getitem__(i)

        # Simple vector selection
        elif isinstance(i, int):
            res = Vector()
            for key, value in self.items():
                res[key] = value[i]
            res.set_uid(self._vectors_uid[i])
            return res

        # Numpy behavior
        else:
            try:
                res = copy(self)
                for key, value in res.items():
                    res[key] = value[i]
                res._vectors_uid = res._vectors_uid[i]
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

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset["x"]
            [[1, 2, 3], [-1, -2, -3]]
            >>> dataset[0]
            Vector({"x":[1,2,3], "y":2})
            >>> dataset[0:2]
            Dataset([
                Vector({"x":[1,2,3], "y":2}),
                Vector({"x":[-1,-2,-3], "y":3}),
            ])
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

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.index(Vector({"x":[1,2,3], "y":2}))
            [0]
        """

        if not isinstance(vector, Vector):
            raise TypeError(f"vector must be a Vector, not {type(Vector)}")
        
        res = []

        for i, v in enumerate(self):
            if v == vector:
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

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3})
            ... ])
            >>> dataset.append(Vector({"x":[9,8,7], "y":4}))
            Dataset([
                Vector({"x":[1,2,3], "y":2}),
                Vector({"x":[-1,-2,-3], "y":3}),
                Vector({"x":[9,8,7], "y":4}),
            ])
        """
        if not isinstance(vector, Vector):
            raise TypeError(f"vector must be a Vector, not {type(vector)}")

        if self.is_empty():

            for label, element in vector.items():
                self[label] = np.array([element])
            self._vectors_uid = np.array([vector.uid])

        else:
            if self.labels != vector.labels:
                raise TypeError(f"vector must have the same keys as the dataset, found vector label: {list(vector.keys())} and dataset labels: {self.labels}")
            
            if self.shapes != vector.shapes:
                raise TypeError(f"vector must have the same shapes as the vectors in the dataset, found vector shapes: {list(vector.shapes)} and dataset shapes: {self.shapes}")

            was_normalized = self.is_normalized()
            self.denormalize()

            for label, element in vector.items():
                self[label] = np.append(self[label], [element], axis=0)
            self._vectors_uid = np.append(self._vectors_uid, [vector.uid], axis=0)

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

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2})
            ... ])
            >>> dataset.add([
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            Dataset([
                Vector({"x":[1,2,3], "y":2}),
                Vector({"x":[-1,-2,-3], "y":3}),
                Vector({"x":[9,8,7], "y":4}),
            ])
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
    
    # Iteration ---------------------------------------------------------------

    def __iter__(self):
        """Iterate over the dataset

        Yields:
            Vector: the next vector in the dataset
        
        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> for vector in dataset:
            ...     print(vector)
            Vector({"x":[1,2,3], "y":2})
            Vector({"x":[-1,-2,-3], "y":3})
            Vector({"x":[9,8,7], "y":4})
        """

        for i in range(len(self)):
            yield self[i]

    # Remove vectors ----------------------------------------------------------

    def pop(self, i:int):
        """Remove a vector from the dataset and return it

        Args:
            i (int): the index of the vector to remove
        
        Returns:
            Vector: the removed vector

        Raises:
            TypeError: if i is not an int

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.pop(1)
            Vector({"x":[-1,-2,-3], "y":3})
            >>> dataset
            Dataset([
                Vector({"x":[1,2,3], "y":2}),
                Vector({"x":[9,8,7], "y":4}),
            ])
        """
        if not isinstance(i, int):
            raise TypeError(f"i must be an int, not {type(i)}")

        was_normalized = self.is_normalized()
        self.denormalize()

        popped = Vector()
        for label, column in self.items():
            popped[label] = column[i]
            self[label] = np.delete(column, i, axis=0)

        if was_normalized:
            self.normalize()

        return popped

    def remove(self, i:Union[int, Vector, List[int], List[Vector]]):
        """Remove a vector from the dataset

        Args:
            i (Union[int, Vector, List[int], List[Vector]]): the index or the vector to remove

        Raises:
            TypeError: if i is not an int, a Vector, a list of int or a list of Vector

        Returns:
            Dataset: the dataset

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.remove(1)
            Dataset([
                Vector({"x":[1,2,3], "y":2}),
                Vector({"x":[9,8,7], "y":4}),
            ])
        """
    
        # if not isinstance(i, int) or not isinstance(i, Vector):
        #     i = [i]
        
        # for j in i:
        #     if not isinstance(j, int) or not isinstance(j, Vector):
        #         raise TypeError(f"i must be an int, a Vector, a list of int or a list of Vector, not {type(i)}")
        
        # TODO
        raise NotImplementedError

        return self      

    # Get info about dataset shape --------------------------------------------

    def is_empty(self):
        """Check if the dataset is empty

        Returns:
            bool: True if the dataset is empty, False otherwise

        Examples:
            >>> dataset = Dataset()
            >>> dataset.is_empty()
            True
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ... ])
            >>> dataset.is_empty()
            False
        """

        if len(self.keys()) == 0:
            return True
        
        if len(self[0]) == 0:
            return True
        
        return False

    @property
    def lines(self):
        """Get the number of vector in the dataset
        
        Returns:
            int: the number of vector in the dataset

        Alias:
            len

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.lines
            3
        """
        if self.is_empty():
            return 0
        return list(self.values())[0].shape[0]
    
    def __len__(self):
        """Get the number of vector in the dataset
        
        Returns:
            int: the number of vector in the dataset

        Alias:
            lines

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> len(dataset)
            3
        """
        return self.lines
    
    @property
    def columns(self):
        """Get the number of vector in the dataset
        
        Returns:
            int: the number of vector in the dataset

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.columns
            2
        """
        return super().__len__()

    @property
    def shape(self):
        """Get the shape of the dataset

        Returns:
            tuple[int]: the shape of the dataset

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.shape
            (3, 2)
        """
        return self.lines, self.columns
    
    @property
    def size(self):
        """Get the size of the dataset

        Returns:
            int: the size of the dataset

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.size
            6
        """
        return self.lines * self.columns
    
    # Get field labels --------------------------------------------------------
    
    @property
    def labels(self):
        """Get the labels
        
        Returns:
            list[str]: the labels

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.labels
            ["x", "y"]
        """
        return list(self.keys())
    
    # Get field shapes --------------------------------------------------------
    
    @property
    def shapes(self):
        """Get the shape of each field
        
        Returns:
            list[tuple[int]]: the shape of each field

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.shapes
            {'x':(3,), 'y':()}
        """
        if self.is_empty():
            return {}
        return self[0].shapes

    @property
    def sizes(self):
        """Get the size of each field
        
        Returns:
            list[int]: the size of each field

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.sizes
            {'x':3, 'y':1}
        """
        return self[0].sizes

    # Get normalization infos -------------------------------------------------

    def is_normalized(self):
        """Check if the dataset is normalized

        Returns:
            bool: True if the dataset is normalized, False otherwise

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.is_normalized()
            False
            >>> dataset.normalize()
            >>> dataset.is_normalized()
            True
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

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.mean("x")
            2.6666666666666665
            >>> dataset.mean()
            {'x': 2.6666666666666665, 'y': 3.0}
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

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.std("x")
            4.189935029992179
            >>> dataset.std()
            {'x': 4.189935029992179, 'y': 0.816496580927726}
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

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.max("x")
            9
            >>> dataset.max()
            {'x': 9, 'y': 4}
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

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.min("x")
            -3
            >>> dataset.min()
            {'x': -3, 'y': 2}
        """
        if field is not None:
            if not isinstance(field, str):
                raise TypeError(f"field must be a string, not {type(field)}")
            return self._mins[field]
        return self._mins

    # Dataset normalization ---------------------------------------------------

    def normalize(self, method:str="zscore"):
        """Normalize the dataset

        Args:
            method (str): normalization method: "zscore" or "minmax"

        Returns:
            Dataset: the normalized dataset

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.normalize()
            Dataset([
                Vector({"x":[-0.39777864, -0.15911146,  0.07955573], "y":-1.22474487}),
                Vector({"x":[-0.87511301, -1.1137802 , -1.35244738], "y":0.}),
                Vector({"x":[ 1.03422447,  1.27289165,  1.51155884], "y":1.22474487}),
            ])
            >>> dataset.normalize("min-max")
            Dataset([
                Vector({'x': [0.33333333, 0.41666667, 0.5       ], 'y': 0.}),
                Vector({'x': [O.16666667, 0.08333333, 0.        ], 'y': 0.5}),
                Vector({'x': [0.83333333, 0.91666667, 1.        ], 'y': 1.})
            ])
        """

        res = copy(self)

        res._normalization_method = method

        if res.is_normalized():
            return
        
        for key, value in res.items():

            res._means[key] = value.mean()
            res._stds[key] = value.std()
            res._maxs[key] = value.max()
            res._mins[key] = value.min()

            if method == "zscore":
                std = value.std()
                if std == 0:
                    std = 1
                res[key] = (value - value.mean()) / std
            elif method == "minmax":
                delta = value.max() - value.min()
                if delta == 0:
                    delta = 1
                res[key] = (value - value.min()) / delta
            else:
                raise ValueError(f"Unknown method: {res._normalization_method}. Accepting only 'zscore' and 'minmax'.")

        res._normalized = True

        return res

    def denormalize(self, label=None, value=None):
        """Denormalize the dataset, a colummn if the label is specified, or a given value if the label and the value are specified

        Raises:
            ValueError: if the label is not specified and the value is specified
            ValueError: if the dataset is not normalized and the value is specified
            TypeError: if the label is not a string or None
        
        Returns:
            Dataset | numpy.ndarray | object: the denormalized dataset | column | value

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.normalize()
            >>> dataset.denormalize("x")
            [[1,2,3],[-1,-2,-3],[9,8,7]]
            >>> dataset.denormalize("x", 0.5)
            4.761634181662756
            >>> dataset.denormalize()
            Dataset([
                Vector({"x":[1,2,3], "y":2}),
                Vector({"x":[-1,-2,-3], "y":3}),
                Vector({"x":[9,8,7], "y":4}),
            ])
        """

        if label is None and value is not None:
            raise ValueError("Cannot denormalize a value without a label")
        if not self.is_normalized() and value is not None:
            raise ValueError("not normalized dataset cannot be be used to denormalize a value")
        if label is not None and not isinstance(label, str):
            raise TypeError(f"label must be a string, not {type(label)}")

        if label is not None:        
            if value is None:
                value = copy(self[label])
            if self._normalization_method == "zscore":
                return value * self._stds[label] + self._means[label]
            elif self._normalization_method == "minmax":
                return value * (self._maxs[label] - self._mins[label]) + self._mins[label]
            else:
                raise ValueError(f"Dataset was normalized with an unknown methode: {self._normalization_method}")

        if not self.is_normalized():
            return self
        
        res = copy(self)
        
        for label, value in res.items():
            if res._normalization_method == "zscore":
                res[label] = value * res._stds[label] + res._means[label]
            elif res._normalization_method == "minmax":
                res[label] = value * (res._maxs[label] - res._mins[label]) + res._mins[label]
            else:
                raise ValueError(f"Dataset was normalized with an unknown methode: {res._normalization_method}.")

        res._normalized = False

        return res
    
    # Shuffle -----------------------------------------------------------------

    def shuffle(self, verbose=False):
        """Shuffle the dataset
        
        Returns:
            Dataset: the shuffled dataset

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.shuffle()
            Dataset([
                Vector({"x":[9,8,7], "y":4}),
                Vector({"x":[1,2,3], "y":2}),
                Vector({"x":[-1,-2,-3], "y":3}),
            ])
        """

        if verbose:
            logs.info(f"Shuffling {self.name}'s Dataset...")

        # Get random indices
        idx = np.random.permutation(len(self))

        # Shuffle the dataset using the randomized indices
        shuffled_dataset = self[idx]

        if verbose:
            logs.info(f"{self.name} dataset shuffled ✅")

        return shuffled_dataset

    # Split dataset -----------------------------------------------------------

    def split(self, val:float, test:float=0.0):
        """Split the dataset into train, val and test subsets.
        
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

        Examples:
            >>> dataset = Dataset([
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> train, val, test = dataset.split(val=1/3, test=1/3)
            >>> train
            Dataset([
                Vector({"x":[9,8,7], "y":4}),
            ])
            >>> val
            Dataset([
                Vector({"x":[-1,-2,-3], "y":3}),
            ]) 
            >>> test
            Dataset([
                Vector({"x":[1,2,3], "y":2}),
            ])
        """

        if not isinstance(val, float) or not isinstance(test, float):
            raise TypeError("val and test must be floats")

        if 0 >= val + test >= 1:
            raise ValueError("val + test must be between 0 and 1")

        index = np.arange(len(self))

        test_N = int(len(self) * test)
        test_step = int(len(self) / test_N)

        test_mask = (index % test_step) == 0

        test_set = self[test_mask]

        rest = self[~test_mask]

        index = np.arange(len(rest))
        mask = np.ones(len(rest), dtype=bool)

        val_N = int(len(rest) * val / (1-test))
        val_step = int(len(rest) / val_N)
        val_mask = mask * ((index % val_step) == 0)

        rest = rest.shuffle()
        val_set = rest[val_mask]
        train_set = rest[~val_mask]

        return train_set, val_set, test_set

    # Get string representation -----------------------------------------------

    def __str__(self) -> str:
        """Get data summary as str

        Returns:
            str: the data summary

        Examples:
            >>> dataset = Dataset(name="MyData", vectors=[
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.normalize()
            >>> dataset.split(val=1/3, test=1/3)
            >>> print(dataset)
            MyData dataset, containing 3 vectors.
            Subsets: Train: 1 vectors, Val: 1 vectors, Test: 1 vectors.

            Fields:
                - x:   Mean= 2.67e+00   Std= 4.19e+00   Min=-3.00e+00   Max= 9.00e+00   Shape=(3,)
                - y:   Mean= 3.00e+00   Std= 0.81e+00   Min= 2.00e+00   Max= 4.00e+00   Shape=()
        """
        return self.__repr__()
    
    def __repr__(self) -> str:
        """Get printable-ready data summary
        
        Returns:
            str: the data summary

        Examples:
            >>> dataset = Dataset(name="MyData", vectors=[
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.normalize()
            >>> dataset.split(val=1/3, test=1/3)
            >>> print(dataset)
            MyData dataset, containing 3 vectors.
            Subsets: Train: 1 vectors, Val: 1 vectors, Test: 1 vectors.

            Fields:
                - x:   Mean= 2.67e+00   Std= 4.19e+00   Min=-3.00e+00   Max= 9.00e+00   Shape=(3,)
                - y:   Mean= 3.00e+00   Std= 0.81e+00   Min= 2.00e+00   Max= 4.00e+00   Shape=()
        """

        if self.lines == 0:
            return f"{self.name} dataset, containing 0 vectors."

        max_label_length = max([len(i) for i in self.labels])
        
        res = f"{self.name} dataset, containing {len(self)} vectors."
        
        res += "\nFields:"

        for label in self.labels:
            res += f"\n   - {label + ' ' * (max_label_length - len(label))}"

            m = self[label].mean()
            s = self[label].std()
            mi = self[label].min()
            ma = self[label].max()
        
            res += f"   Mean: {(' ' if m >= 0 else '')}{m:.2e}"
            res += f"   Std: {(' ' if s >= 0 else '')}{s:.2e}"
            res += f"   Min: {(' ' if mi >= 0 else '')}{mi:.2e}"
            res += f"   Max: {(' ' if ma >= 0 else '')}{ma:.2e}"
            res += f"   Shape: {self.shapes[label]}"

        return res
    
    def raw_str(self):
        """Get a string representation of the dataset fields as a dictionary

        Returns:
            str: String representation of dataset fields

        Example:
            >>> dataset = Dataset(name="MyData", vectors=[
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.raw_str()
            {'x':[[1,2,3],[-1,-2,-3],[9,8,7]], 'y':[2,3,4]}
        """       

        return super().__repr__()

    # Print few vectors -------------------------------------------------------

    def print_few_vectors(self, count=5) -> str:
        """Print a resume of few vectors in the dataset
        
        Args:
            count (int, optional): number of vectors to print. Defaults to 5.

        Returns:
            str: the resume that is printed

        Examples:
            >>> dataset = Dataset(name="MyData", vectors=[
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.print_few_vectors(2)
            Here is 2 vectors from the MyData dataset:
            Vector 1
                - x:   Mean=-2.00e+00   Std= 0.82e+00   Min=-3.00e+00   Max=-1.00e+00   Shape=(3,)
                - y:   Value= 3.00e+00
            Vector 2
                - x:   Mean= 8.00e+00   Std= 0.82e+00   Min= 7.00e+00   Max= 9.00e+00   Shape=(3,)
                - y:   Value= 4.00e+00
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

        Examples:
            >>> dataset = Dataset(name="MyData", vectors=[
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset.filter("x")
            MyData dataset, containing 3 vectors. (Not processed, no statistics available)

            Fields:
                - x   Mean=-2.00e+00   Std= 0.82e+00   Min=-3.00e+00   Max= 9.00e+00   Shape=(3,)
        """

        if verbose:
            logs.info(f"Filtering {self.name} dataset...")

        if isinstance(labels, str):
            labels = [labels]

        filtered = copy(self)
        
        for label in self.labels:
            if not isinstance(label, str):
                raise TypeError("Labels must be strings")
            
            if label not in labels:
                del filtered[label]

                if filtered.is_normalized():
                    del filtered._means[label]
                    del filtered._stds[label]
                    del filtered._mins[label]
                    del filtered._maxs[label]

        if verbose:
            logs.info(f"{self.name} dataset filtered ✅\n{filtered}")

        if not new_name:
            filtered.name = f"{self.name} filtered"

        return filtered

    def __eq__(self, other):
        """Check if two datasets are equal

        Args:
            other (Dataset): the other dataset

        Returns:
            bool: True if the datasets are equal, False otherwise

        Examples:
            >>> dataset1 = Dataset(name="MyData", vectors=[
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset2 = Dataset(name="MyData", vectors=[
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset1 == dataset2
            True
        """

        if not isinstance(other, Dataset):
            return False

        if len(self) != len(other):
            return False

        for i in range(len(self)):
            if self[i] != other[i]:
                return False

        return True
    
    def __ne__(self, other):
        """Check if two datasets are different

        Args:
            other (Dataset): the other dataset

        Returns:
            bool: True if the datasets are different, False otherwise

        Examples:
            >>> dataset1 = Dataset(name="MyData", vectors=[
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset2 = Dataset(name="MyData", vectors=[
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset3 = Dataset(name="MyData", vectors=[
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[-1,-2,-3], "y":3}),
            ...     Vector({"x":[9,8,7], "y":5}),
            ... ])
            >>> dataset1 != dataset2
            False
            >>> dataset1 != dataset3
            True
        """

        return not self.__eq__(other)

    def approx(self, other, tol=1e-14):
        """Check if two datasets are approximately equal

        Args:
            other (Dataset): the other dataset
            tol (float, optional): the tolerance. Defaults to 1e-14.

        Returns:
            bool: True if the datasets are approximately equal, False otherwise

        Examples:
            >>> dataset1 = Dataset(name="MyData", vectors=[
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[9,8,7], "y":4}),
            ... ])
            >>> dataset2 = Dataset(name="MyData", vectors=[
            ...     Vector({"x":[1,2,3], "y":2}),
            ...     Vector({"x":[9,8,7], "y":5}),
            ... ])
            >>> dataset1.approx(dataset2, tol=1e-10))
            False
            >>> dataset1.approx(dataset3, tol=10)
            True
        """

        if not isinstance(other, Dataset):
            return False

        if self.lines != other.lines:
            return False
        
        if self.columns != other.columns:
            return False
        
        if self.shapes != other.shapes:
            return False

        for i in range(len(self)):
            if not self[i].approx(other[i], tol=tol):
                return False

        return True

    
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

    def get_v0():
        return Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]], z=2)
    
    def get_v1():
        return Vector(x=[3, 1, -2], y=[[-4, -5, 3],[6, 2, -1]], z=-3)
    
    def get_v2():
        return Vector(x=[-2, 3, 1], y=[[2, 3, -4],[-5, 1, 6]], z=4)
    
    def get_dataset():
        return Dataset([TestDataset.get_v0(), TestDataset.get_v1(), TestDataset.get_v2()], name="MyData")
        
    def test_constructor(self):
        v0 = TestDataset.get_v0()
        v1 = TestDataset.get_v1()
        v2 = TestDataset.get_v2()

        dataset = Dataset([v0, v1, v2])

    def test_name(self):
        dataset = TestDataset.get_dataset()
        
        assert dataset.name == "MyData"

        dataset.name = "MyNewData"
        assert dataset.name == "MyNewData"

    def test_len(self):
        dataset = Dataset([TestDataset.get_v0(), TestDataset.get_v1(), TestDataset.get_v1()])
        assert len(dataset) == 3

    def test_getitem(self):
        dataset = TestDataset.get_dataset()

        assert dataset[0] == TestDataset.get_v0(), dataset[0]
        assert dataset[1] == TestDataset.get_v1(), dataset[1]
        assert dataset[2] == TestDataset.get_v2(), dataset[2]

        assert dataset["x"].tolist() == [[1, -2, 3],[3, 1, -2], [-2, 3, 1]], dataset["x"].tolist()
        assert dataset["y"].tolist() == [[[-4, -5, 6],[7, 8, -9]], [[-4, -5, 3],[6, 2, -1]], [[2, 3, -4],[-5, 1, 6]]], dataset["y"].tolist()
        assert dataset["z"].tolist() == [2, -3, 4], dataset["z"].tolist()

        assert dataset[:2] == Dataset([TestDataset.get_v0(), TestDataset.get_v1()]), dataset[:2]

        assert dataset.get(0) == dataset[0]
        assert dataset.get("x").tolist() == dataset["x"].tolist()
        assert dataset.get([0,1]) == dataset[:2]

    def test_index(self):
        dataset = TestDataset.get_dataset()

        assert dataset.index(TestDataset.get_v0()) == [0], dataset.index(TestDataset.get_v0())
        assert dataset.index(TestDataset.get_v1()) == [1], dataset.index(TestDataset.get_v1())
        assert dataset.index(TestDataset.get_v2()) == [2], dataset.index(TestDataset.get_v2())

    def test_normalize(self):
        dataset = TestDataset.get_dataset()

        dataset = dataset.normalize()

        eps = 1e-14

        assert 0-eps < dataset["x"].mean() < 0+eps, dataset["x"].mean()
        assert 1-eps < dataset["x"].std() < 1+eps, dataset["x"].std()
        assert 0-eps < dataset["y"].mean() < 0+eps, dataset["y"].mean()
        assert 1-eps < dataset["y"].std() < 1+eps, dataset["y"].std()
        assert 0-eps < dataset["z"].mean() < 0+eps, dataset["z"].mean()
        assert 1-eps < dataset["z"].std() < 1+eps, dataset["z"].std()

        dataset = TestDataset.get_dataset()

        dataset = dataset.normalize("minmax")

        assert dataset["x"].min() == 0, dataset["x"].min()
        assert dataset["x"].max() == 1, dataset["x"].max()
        assert dataset["y"].min() == 0, dataset["y"].min()
        assert dataset["y"].max() == 1, dataset["y"].max()
        assert dataset["z"].min() == 0, dataset["z"].min()
        assert dataset["z"].max() == 1, dataset["z"].max()

    def test_denormalize(self):
        dataset = TestDataset.get_dataset()
        dataset2 = TestDataset.get_dataset()

        dataset = dataset.normalize()
        dataset = dataset.denormalize()

        assert dataset.approx(dataset2), f"1) {dataset}\n\n2) {dataset2}"

        dataset = TestDataset.get_dataset()
        dataset = dataset.normalize("minmax")
        dataset = dataset.denormalize()

        assert dataset.approx(dataset2), f"1) {dataset}\n\n2) {dataset2}"

    def test_append(self):
        dataset = Dataset([TestDataset.get_v0(), TestDataset.get_v1()])
        assert len(dataset) == 2, len(dataset)
        dataset.append(TestDataset.get_v2())
        assert len(dataset) == 3, len(dataset)
        assert dataset[2] == TestDataset.get_v2(), dataset[2]

    def test_pop(self):
        dataset = TestDataset.get_dataset()

        assert dataset.pop(0) == TestDataset.get_v0()
        assert len(dataset) == 2, len(dataset)
        assert dataset.pop(1) == TestDataset.get_v2()
        assert len(dataset) == 1, len(dataset)
        assert dataset.pop(0) == TestDataset.get_v1()
        assert len(dataset) == 0, len(dataset)

    # def test_remove(self):

    #     return

    #     dataset = TestDataset.get_dataset()

    #     dataset.remove(TestDataset.get_v0())
    #     assert len(dataset) == 2
    #     dataset.remove(TestDataset.get_v2())
    #     assert len(dataset) == 1
    #     dataset.remove(TestDataset.get_v1())
    #     assert len(dataset) == 0
        
    #     dataset = TestDataset.get_dataset()

    #     dataset.remove(0)
    #     assert len(dataset) == 2
    #     dataset.remove(1)
    #     assert len(dataset) == 1
    #     dataset.remove(0)
    #     assert len(dataset) == 0

    #     dataset = TestDataset.get_dataset()

    #     dataset.remove([0,2])
    #     assert len(dataset) == 1

    #     dataset = TestDataset.get_dataset()

    #     dataset.remove([TestDataset.get_v0(), TestDataset.get_v2()])
    #     assert len(dataset) == 1

    def test_lines(self):
        dataset = TestDataset.get_dataset()

        assert dataset.lines == 3
        assert len(dataset) == 3

    def test_columns(self):
        dataset = TestDataset.get_dataset()

        assert dataset.columns == 3

    def test_shape(self):
        dataset = TestDataset.get_dataset()

        assert dataset.shape == (3,3)
        assert dataset.shapes == {"x": (3,), "y": (2,3), "z": ()}

    def test_size(self):
        dataset = TestDataset.get_dataset()
        
        assert dataset.size == 9, dataset.size
        assert dataset.sizes == {"x": 3, "y": 6, "z": 1}, dataset.sizes

    def test_labels(self):
        dataset = TestDataset.get_dataset()

        assert dataset.labels == ["x", "y", "z"]

    def test_shuffle(self):
        dataset = TestDataset.get_dataset()
        dataset.append(TestDataset.get_v0()*2)
        dataset.append(TestDataset.get_v1()*2)
        dataset.append(TestDataset.get_v2()*2)
        dataset.append(TestDataset.get_v0()*-1)
        dataset.append(TestDataset.get_v1()*-1)
        dataset.append(TestDataset.get_v2()*-1)
        dataset.append(TestDataset.get_v0()*0)

        copy_dataset = copy(dataset)

        dataset = dataset.shuffle()

        # ⚠️ As shuffle is fully random, it is possible that the shuffle gives the same order as the original dataset and then make a false positive

        assert dataset.index(TestDataset.get_v0()) != []
        assert dataset.index(TestDataset.get_v1()) != []
        assert dataset.index(TestDataset.get_v2()) != []
        assert dataset.index(TestDataset.get_v0()*2) != []
        assert dataset.index(TestDataset.get_v1()*2) != []
        assert dataset.index(TestDataset.get_v2()*2) != []
        assert dataset.index(TestDataset.get_v0()*-1) != []
        assert dataset.index(TestDataset.get_v1()*-1) != []
        assert dataset.index(TestDataset.get_v2()*-1) != []
        assert dataset.index(TestDataset.get_v0()*0) != []
        assert dataset != copy_dataset, f"\n\nOriginal:\n{copy_dataset.raw_str()}\n\nShuffled: {dataset.raw_str()}"
        assert np.any(dataset._vectors_uid != copy_dataset._vectors_uid), f"\nOriginal: {copy_dataset._vectors_uid}\nShuffled: {dataset._vectors_uid}"
        assert np.all(np.sort(dataset._vectors_uid) == np.sort(copy_dataset._vectors_uid)), f"\nOriginal: {np.sort(copy_dataset._vectors_uid)}\nShuffled: {np.sort(dataset._vectors_uid)}"

        assert len(dataset) == 10, len(dataset)

    def test_split(self):
        dataset = TestDataset.get_dataset()

        dataset.append(TestDataset.get_v0()*2)
        dataset.append(TestDataset.get_v1()*2)
        dataset.append(TestDataset.get_v2()*2)
        dataset.append(TestDataset.get_v0()*-1)
        dataset.append(TestDataset.get_v1()*-1)
        dataset.append(TestDataset.get_v2()*-1)
        dataset.append(TestDataset.get_v0()*0)

        train, val, test = dataset.split(0.3, 0.1)

        assert len(train) == 6, len(train)
        assert len(val) == 3, len(val)
        assert len(test) == 1, len(test)

    def test_filter(self):
        dataset = TestDataset.get_dataset()

        new_dataset = dataset.filter(['x','y'])

        assert len(new_dataset) == 3
        assert new_dataset.labels == ['x','y']

if __name__ == '__main__':
    unittest.main()