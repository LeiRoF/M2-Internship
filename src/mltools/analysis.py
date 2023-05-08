from dataclasses import dataclass
from typing import List, Dict
import numpy as np

#==============================================================================
# RESULT
#==============================================================================

@dataclass
class Result:
    expected: np.ndarray
    predictions: List[np.ndarray]

    # Accessors ---------------------------------------------------------------

#==============================================================================
# MODEL
#==============================================================================

@dataclass
class Model:
    results: Dict[str, Result]

    # Accessors ---------------------------------------------------------------

    def get(self, label:str) -> Result:
        return self.results[label]
    
    def filter(self, labels:list[str]) -> "Model":
        return Model(
            results={label: self.results[label] for label in labels}
        )

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
