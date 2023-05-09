from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import unittest

#==============================================================================
# RESULT
#==============================================================================

@dataclass
class Result:
    _index: int
    _expected: np.ndarray
    _predictions: List[np.ndarray]

    # Accessors ---------------------------------------------------------------

    @property
    def expected(self) -> np.ndarray:
        """The expected values
        
        Returns:
            np.ndarray: The expected values
        """
        return self._expected
    
    @property
    def predictions(self) -> List[np.ndarray]:
        """The predictions

        Returns:
            List[np.ndarray]: The predictions
        """
        return self._predictions

# Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Test_Result(unittest.TestCase):
    ... # TODO

#==============================================================================
# MODEL
#==============================================================================

@dataclass
class Model:
    _id:int
    _results: Dict[str, Result]

    # Accessors ---------------------------------------------------------------

    def get(self, label:str) -> Result:
        """Get a result of this model
     
        Args:
            label (str): The label of the result to get

        Returns:
            Result: The result
        """
        return self.results[label]
    
    def filter(self, labels:list[int]) -> "Model":
        """Filter the results of this model

        Args:
            labels (list[int]): The labels to keep

        Returns:
            Model: The filtered model
        """
        return Model(
            results={label: self.results[label] for label in labels}
        )
    
# Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Test_Model(unittest.TestCase):
    ... # TODO

#==============================================================================
# MODEL SET
#==============================================================================

@dataclass
class ModelSet:
    models: Dict[str, Model]

    # Accessors ---------------------------------------------------------------

    def get(self, label:str) -> Model:
        return self.models[label]

    def filter(self, labels:list[str]) -> "ModelSet":
        return ModelSet(
            models={label: self.models[label] for label in labels}
        )

# Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Test_ModelSet(unittest.TestCase):
    ... # TODO