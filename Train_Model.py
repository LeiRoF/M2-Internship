
from src.mltools.dataset import Dataset
import numpy as np
from typing import Union

print("\nImporting tensorflow --------------------------------------------------------\n")
import tensorflow as tf
mae = tf.keras.metrics.MeanAbsoluteError(name="MAE")
print("\nEnd importing tensorflow --------------------------------------------------------\n")


#==============================================================================
# CONFIGURATION
#==============================================================================

valid_frac = 0.2
test_frac  = 0.1
raw_dataset_path = "data/dataset" # path to the raw dataset
dataset_archive = "data/dataset.npz" # path to the dataset archive (avoid redoing data processing)
epochs = 100
batch_size=100
loss = 'mean_squared_error'
optimizer = 'SGD'
metrics = [
    mae,
]

#==============================================================================
# LOAD DATASET
#==============================================================================

def data_loader(path:str):
    """Load a dataset element"""

    data = np.load(path)

    x = {
        "Dust wavelenght": np.array([250.,]), # [um]
        "Dust map" :       data["dust_image"].reshape(*data["dust_image"].shape, 1), # adding a channel dimension
        "CO velocity" :    data["CO_v"],
        "CO cube" :        data["CO_cube"].reshape(*data["CO_cube"].shape, 1), # adding a channel dimension
        "N2H+ velocity" :  data["N2H_v"],
        "N2H cube" :       data["N2H_cube"].reshape(*data["N2H_cube"].shape, 1), # adding a channel dimension
    }
    
    y = {
        "Total mass" :      np.array(data["mass"]),
        "Max temperature" : np.array(np.amax(data["dust_temperature"])),
    }

    return x, y

dataset = Dataset("Pre stellar cores", loader=data_loader, raw_path=raw_dataset_path, archive_path=dataset_archive)