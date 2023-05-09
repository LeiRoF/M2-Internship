import numpy as np
from typing import Union, Callable
import os
from LRFutils import progress, logs
import sysinfo
import matplotlib.pyplot as plt
from copy import deepcopy as copy
from multiprocessing import Pool

from typing import Union, Callable, List, Dict, Tuple, Any

class Vector(Dict):
    def __init__(self, **kwargs):
        """Vector class

        Args:
            **kwargs (dict): Dictionary of vectors

        Raises:
            TypeError: If any of the values is not convertible to a numpy array

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]], z=2)
            >>> print(v)
            Vector(
                x: Mean= 6.66e-01   Std= 2.05e+00   Min=-2.00e+00   Max= 3.00e+00   Shape=(3,)
                y: Mean= 0.50e-01   Std= 6.60e+00   Min=-9.00e+00   Max= 8.00e+00   Shape=(2, 3)
                z: Value= 2.00e+00
            )
        """

        super().__init__()

        for key, value in kwargs.items():
            try:
                self[key] = np.array(value)
            except:
                raise TypeError(f"{key} must be convertible to a numpy array")

    def __add__(self, element):
        """Add a vector to anything

        Args:
            element (eny): element to add

        Returns:
            Vector: Vector sum

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]], z=2)
            >>> v2 = 2 + v
            >>> print(v2["x"])
            [3, 0, 5]
        """

        if isinstance(element, Vector):
            return Vector(**{key: self[key] + element[key] for key in self.keys()})
        else:
            return Vector(**{key: self[key] + element for key in self.keys()})
    
    def __radd__(self, element):
        """Add a vector to anything

        Args:
            vector (Vector): Vector to add

        Returns:
            Vector: Vector sum

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]], z=2)
            >>> v2 = 2 + v
            >>> print(v2["x"])
            [3, 0, 5]
        """

        return self + element
    
    def __sub__(self, element):
        """Subtract a vector from anything

        Args:
            element (any): element to subtract

        Returns:
            Vector: Vector difference

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]], z=2)
            >>> v2 = v - 2
            >>> print(v2["x"])
            [-1, -4, 1]
        """

        if isinstance(element, Vector):
            return Vector(**{key: self[key] - element[key] for key in self.keys()})
        else:
            return Vector(**{key: self[key] - element for key in self.keys()})
    
    def __rsub__(self, element):
        """Subtract a vector from anything

        Args:
            element (any): element to subtract

        Returns:
            Vector: Vector difference

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]], z=2)
            >>> v2 = 2 - v
            >>> print(v2["x"])
            [1, 4, -1]
        """

        if isinstance(element, Vector):
            return Vector(**{key: element[key] - self[key] for key in self.keys()})
        else:
            return Vector(**{key: element - self[key] for key in self.keys()})
    
    def __mul__(self, element):
        """Multiply a vector by anything

        Args:
            element (any): element to multiply

        Returns:
            Vector: Vector product

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]], z=2)
            >>> v2 = v * 2
            >>> print(v2["x"])
            [2, -4, 6]
        """

        if isinstance(element, Vector):
            return Vector(**{key: self[key] * element[key] for key in self.keys()})
        else:
            return Vector(**{key: self[key] * element for key in self.keys()})
    
    def __rmul__(self, element):
        """Multiply a vector by anything

        Args:
            element (any): element to multiply

        Returns:
            Vector: Vector product

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]], z=2)
            >>> v2 = 2 * v
            >>> print(v2["x"])
            [2, -4, 6]
        """
    
        return self * element
    
    def __truediv__(self, element):
        """Divide a vector

        Args:
            element (any): denominator

        Raises:
            TypeError: If vector is not a Vector object

        Returns:
            Vector: Vector quotient

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]], z=2)
            >>> v2 = v / 2
            >>> print(v2["x"])
            [0.5, -1., 1.5]
            >>> print(v2["y"])
            [[-2., -2.5, 3.],[3.5, 4., -4.5]]
            >>> print(v2["z"])
            1.0
        """

        if isinstance(element, Vector):
            return Vector(**{key: self[key] / element[key] for key in self.keys()})
        else:
            return Vector(**{key: self[key] / element for key in self.keys()})
    
    def __rtruediv__(self, element):
        """Divide by a vector

        Args:
            element (any): numerator

        Returns:
            Vector: Vector quotient

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]], z=2)
            >>> v2 = 1 / v
            >>> print(v2["x"])
            [1., -0.5, 0.33333333]
            >>> print(v2["y"])
            [[-0.25, -0.2, 0.16666667],[0.14285714, 0.125, -0.11111111]]
            >>> print(v2["z"])
            0.5
        """

        if isinstance(element, Vector):
            return Vector(**{key: vector[key] / self[key] for key in self.keys()})
        else:
            return Vector(**{key: element / self[key] for key in self.keys()})

    def __getitem__(self, key):
        """Get a vector item

        Args:
            key (str): Vector item key

        Returns:
            np.ndarray: Vector item

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]], z=2)
            >>> print(v["x"])
            [1, -2, 3]
        """

        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        """Set a vector item

        Args:
            key (str): Vector item key
            value (np.ndarray): Vector item value

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]], z=2)
            >>> v["x"] = 0
            >>> print(v)
            Vector(
                x: Value= 0.00e+00
                y: Mean= 0.50e-01   Std= 6.60e+00   Min=-9.00e+00   Max= 8.00e+00   Shape=(2, 3)
                z: Value= 2.00e+00
            )
        """

        super().__setitem__(key, np.array(value))
    
    def __delitem__(self, key):
        """Delete a vector item

        Args:
            key (str): Vector item key

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]], z=2)
            >>> del v["x"]
            >>> print(v)
            Vector(
                y: Mean= 0.50e-01   Std= 6.60e+00   Min=-9.00e+00   Max= 8.00e+00   Shape=(2, 3)
                z: Value= 2.00e+00
            )
        """

        super().__delitem__(key)

    def __deepcopy__(self, memo):
        """Deepcopy a vector

        Args:
            memo (dict): Memoization dictionary

        Returns:
            Vector: Copied vector

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]], z=2)
            >>> v_copy = copy(v)
            >>> v_copy["x"] = 0
            >>> print(v)
            Vector(
                x: Mean= 6.66e-01   Std= 2.05e+00   Min=-2.00e+00   Max= 3.00e+00   Shape=(3,)
                y: Mean= 0.50e-01   Std= 6.60e+00   Min=-9.00e+00   Max= 8.00e+00   Shape=(2, 3)
                z: Value= 2.00e+00
            )
            >>> print(v_copy)
            Vector(
                x: Value= 0.00e+00
                y: Mean= 0.50e-01   Std= 6.60e+00   Min=-9.00e+00   Max= 8.00e+00   Shape=(2, 3)
                z: Value= 2.00e+00
            )
        """

        return Vector(**{key: np.copy(value) for key, value in self.items()})
    
    def __repr__(self):
        """Vector representation

        Returns:
            str: Vector representation

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]], z=2)
            >>> print(v)
            Vector(
                x: Mean= 6.66e-01   Std= 2.05e+00   Min=-2.00e+00   Max= 3.00e+00   Shape=(3,)
                y: Mean= 0.50e-01   Std= 6.60e+00   Min=-9.00e+00   Max= 8.00e+00   Shape=(2, 3)
                z: Value= 2.00e+00
            )
        """

        res = "Vector(\n   "

        for label in self.labels:

            res += f"{label}:"

            m = np.mean(self[label])
            s = np.std(self[label])
            mi = np.min(self[label])
            ma = np.max(self[label])

            if vector.data[label].size > 1:
                res += f"   Mean={(' ' if m >= 0 else '')}{m:.2e}"
                res += f"   Std={(' ' if s >= 0 else '')}{s:.2e}"
                res += f"   Min={(' ' if mi >= 0 else '')}{mi:.2e}"
                res += f"   Max={(' ' if ma >= 0 else '')}{ma:.2e}"
                res += f"   Shape={vector[label].shape}"
            else:
                res += f"   Value={self.denormalize(label, vector[label])[0]:.2e}"
        
        res += "\n)"

        return res
    
    def __str__(self):
        """Vector string

        Returns:
            str: Vector string

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]], z=2)
            >>> print(v)
            Vector(
                x: Mean= 6.66e-01   Std= 2.05e+00   Min=-2.00e+00   Max= 3.00e+00   Shape=(3,)
                y: Mean= 0.50e-01   Std= 6.60e+00   Min=-9.00e+00   Max= 8.00e+00   Shape=(2, 3)
                z: Value= 2.00e+00
            )
        """

        return self.__repr__()

    def __len__(self):
        """Vector length

        Returns:
            int: Vector length

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]])
            >>> len(v)  
            2
        """

        return len(self.keys())
    
    def __iter__(self):
        """Vector iterator

        Yields:
            str: Vector item key
        
        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]])
            >>> for label, element in v:
            ...     print(label, element)
            x [1, -2, 3]
            y [[-4, -5, 6],[7, 8, -9]]
        """

        for label, element in self.items():
            yield label, element

    def __eq__(self, vector):
        """Test if two vectors are equal
        
        Args:
            vector (Vector): Vector to compare

        Raises:
            TypeError: If vector is not a Vector object

        Returns:
            bool: True if equal, False otherwise

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]])
            >>> v == Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]])
            True
            >>> v == Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -8]])
            False
        """

        if not isinstance(vector, Vector):
            raise TypeError(f"vector must be a Vector object, not {type(vector)}")

        return all([np.array_equal(self[key], vector[key]) for key in self.keys()])
    
    def __ne__(self, vector):
        """Test if two vectors are not equal
        
        Args:
            vector (Vector): Vector to compare

        Raises:
            TypeError: If vector is not a Vector object

        Returns:
            bool: True if not equal, False otherwise

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]])
            >>> v != Vector(x=[2, -1, 3]), y=[[-4, -4, 6],[8, 8, -9]])
            True
            >>> v != Vector(x=[1, -2, 3]), y=[[-4, -5, 6],[7, 8, -9]])
            False
        """

        if not isinstance(vector, Vector):
            raise TypeError(f"vector must be a Vector object, not {type(vector)}")

        return not self.__eq__(vector)
    
    def __lt__(self, vector):
        """Test if a vector is less than another
        
        Args:
            vector (Vector): Vector to compare

        Raises:
            TypeError: If vector is not a Vector object

        Returns:
            bool: True if less than, False otherwise

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]])
            >>> v < Vector(x=[2, -1, 3], y=[[-4, -4, 6],[8, 8, -9]])
            True
            >>> v < Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]])
            False
            >>> v < Vector(x=[0, -3, 3], y=[[-4, -6, 6],[6, 8, -9]])
            False
        """

        if not isinstance(vector, Vector):
            raise TypeError(f"vector must be a Vector object, not {type(vector)}")

        return all([np.all(self[key] < vector[key]) for key in self.keys()])
    
    def __le__(self, vector):
        """Test if a vector is less than or equal to another
        
        Args:
            vector (Vector): Vector to compare

        Raises:
            TypeError: If vector is not a Vector object

        Returns:
            bool: True if less than or equal to, False otherwise

        Example:
            >>> v = Vector(x=[1, -2, 3], y=np.array([[-4, -5, 6],[7, 8, -9]])
            >>> v <= Vector(x=[2, -1, 3], y=np.array([[-4, -4, 6],[8, 8, -9]])
            True
            >>> v <= Vector(x=[1, -2, 3], y=np.array([[-4, -5, 6],[7, 8, -9]])
            True
            >>> v <= Vector(x=[0, -3, 3], y=np.array([[-4, -7, 6],[6, 8, -9]])
            False
        """

        if not isinstance(vector, Vector):
            raise TypeError(f"vector must be a Vector object, not {type(vector)}")

        return all([np.all(self[key] <= vector[key]) for key in self.keys()])
    
    def __gt__(self, vector):
        """Test if a vector is greater (element by element) than another
        
        Args:
            vector (Vector): Vector to compare

        Raises:
            TypeError: If vector is not a Vector object

        Returns:
            bool: True if greater than, False otherwise

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]])
            >>> v > Vector(x=[0, -3, 3], y=[[-4, -7, 6],[6, 8, -9]])
            True
            >>> v > Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]])
            False
            >>> v > Vector(x=[2, -1, 3], y=[[-4, -4, 6],[8, 8, -9]])
            False
        """

        if not isinstance(vector, Vector):
            raise TypeError(f"vector must be a Vector object, not {type(vector)}")

        return all([np.all(self[key] > vector[key]) for key in self.keys()])
    
    def __ge__(self, vector):
        """Test if a vector is greater (element by element) than or equal to another
        
        Args:
            vector (Vector): Vector to compare

        Raises:
            TypeError: If vector is not a Vector object

        Returns:
            bool: True if greater than or equal to, False otherwise

        Example:
            >>> v = Vector(x=[1, -2, 3]), y=[[-4, -5, 6],[7, 8, -9]])
            >>> v >= Vector(x=[0, -3, 3]), y=[[-4, -7, 6],[6, 8, -9]])
            True
            >>> v >= Vector(x=[1, -2, 3]), y=[[-4, -5, 6],[7, 8, -9]])
            True
            >>> v >= Vector(x=[2, -1, 3], y=[[-4, -4, 6],[8, 8, -9]])
            False
        """

        if not isinstance(vector, Vector):
            raise TypeError(f"vector must be a Vector object, not {type(vector)}")

        return all([np.all(self[key] >= vector[key]) for key in self.keys()])

    def __abs__(self):
        """Get the absolute value of a vector

        Returns:
            Vector: Absolute value of vector

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]])
            >>> abs(v)
            Vector(x=[1, 2, 3], y=[[4, 5, 6],[7, 8, 9]])
        """

        return Vector(**{key: np.abs(self[key]) for key in self.keys()})
    
    def shapes(self):
        """Get the shapes of all vector fields

        Returns:
            dict: Vector field shapes

        Example:
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]])
            >>> v.shapes()
            {'x': (3,), 'y': (2, 3)}
        """
    
        return {key: self[key].shape for key in self.keys()}

    def sizes(self):
        """Get the sizes of all vector fields

        Returns:
            dict: Vector field sizes

        Example:    
            >>> v = Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]])
            >>> v.sizes()
            {'x': 3, 'y': 6}
        """

        return {key: self[key].size for key in self.keys()}




"""
██╗   ██╗███╗   ██╗██╗████████╗    ████████╗███████╗███████╗████████╗███████╗
██║   ██║████╗  ██║██║╚══██╔══╝    ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔════╝
██║   ██║██╔██╗ ██║██║   ██║          ██║   █████╗  ███████╗   ██║   ███████╗
██║   ██║██║╚██╗██║██║   ██║          ██║   ██╔══╝  ╚════██║   ██║   ╚════██║
╚██████╔╝██║ ╚████║██║   ██║          ██║   ███████╗███████║   ██║   ███████║
 ╚═════╝ ╚═╝  ╚═══╝╚═╝   ╚═╝          ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚══════╝                                                                  
"""

import unittest

class TestVector(unittest.TestCase):

    def get_v():
        return Vector(x=[1, -2, 3], y=[[-4, -5, 6],[7, 8, -9]], z=2)
    
    def get_v2():
        return Vector(x=[3, 1, -2], y=[[-4, -5, 3],[6, 2, -1]], z=-3)

    def test_constructor(self):
        v = TestVector.get_v()
        
        assert v["x"].tolist() == [1, -2, 3], v["x"].tolist()
        assert v["y"].tolist() == [[-4, -5, 6],[7, 8, -9]], v["y"].tolist()
        assert v["z"].tolist() == 2, v["z"].tolist()

    def test_add(self):
        v = TestVector.get_v()
        v2 = TestVector.get_v2()
    
        res = v + 1
        assert res["x"].tolist() == [2, -1, 4], res["x"].tolist()
        assert res["y"].tolist() == [[-3, -4, 7],[8, 9, -8]], res["y"].tolist()
        assert res["z"].tolist() == 3, res["z"].tolist()

        res = 1 + v
        assert res["x"].tolist() == [2, -1, 4], res["x"].tolist()
        assert res["y"].tolist() == [[-3, -4, 7],[8, 9, -8]], res["y"].tolist()
        assert res["z"].tolist() == 3, res["z"].tolist()

        res = v + v2
        assert res["x"].tolist() == [4, -1, 1], res["x"].tolist()
        assert res["y"].tolist() == [[-8, -10, 9],[13, 10, -10]], res["y"].tolist()
        assert res["z"].tolist() == -1, res["z"].tolist()

    def test_sub(self):
        v = TestVector.get_v()
        v2 = TestVector.get_v2()

        res = v - 1
        assert res["x"].tolist() == [0, -3, 2], res["x"].tolist()
        assert res["y"].tolist() == [[-5, -6, 5],[6, 7, -10]], res["y"].tolist()
        assert res["z"].tolist() == 1, res["z"].tolist()

        res = 1 - v
        assert res["x"].tolist() == [0, 3, -2], res["x"].tolist()
        assert res["y"].tolist() == [[5, 6, -5],[-6, -7, 10]], res["y"].tolist()
        assert res["z"].tolist() == -1, res["z"].tolist()

        res = v - v2
        assert res["x"].tolist() == [-2, -3, 5], res["x"].tolist()
        assert res["y"].tolist() == [[0, 0, 3],[1, 6, -8]], res["y"].tolist()
        assert res["z"].tolist() == 5, res["z"].tolist()

    def test_mul(self):
        v = TestVector.get_v()
        v2 = TestVector.get_v2()

        res = v * 2
        assert res["x"].tolist() == [2, -4, 6], res["x"].tolist()
        assert res["y"].tolist() == [[-8, -10, 12],[14, 16, -18]], res["y"].tolist()
        assert res["z"].tolist() == 4, res["z"].tolist()

        res = 2 * v
        assert res["x"].tolist() == [2, -4, 6], res["x"].tolist()
        assert res["y"].tolist() == [[-8, -10, 12],[14, 16, -18]], res["y"].tolist()
        assert res["z"].tolist() == 4, res["z"].tolist()

        res = v * v2
        assert res["x"].tolist() == [3, -2, -6], res["x"].tolist()
        assert res["y"].tolist() == [[16, 25, 18],[42, 16, 9]], res["y"].tolist()
        assert res["z"].tolist() == -6, res["z"].tolist()

    def test_div(self):
        v = TestVector.get_v()
        v2 = TestVector.get_v2()

        res = v / 2
        assert res["x"].tolist() == [0.5, -1, 1.5], res["x"].tolist()
        assert res["y"].tolist() == [[-2, -2.5, 3],[3.5, 4, -4.5]], res["y"].tolist()
        assert res["z"].tolist() == 1, res["z"].tolist()

        res = 2 / v
        assert res["x"].tolist() == [2, -1, 2/3], res["x"].tolist()
        assert res["y"].tolist() == [[-0.5, -0.4, 2/6],[2/7, 2/8, -2/9]], res["y"].tolist()
        assert res["z"].tolist() == 1, res["z"].tolist()

        res = v / v2
        assert res["x"].tolist() == [1/3, -2, -3/2], res["x"].tolist()
        assert res["y"].tolist() == [[1, 1, 2],[7/6, 4, 9]], res["y"].tolist()
        assert res["z"].tolist() == -2/3, res["z"].tolist()


if __name__ == '__main__':
    unittest.main()