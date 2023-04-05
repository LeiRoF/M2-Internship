import numpy as np
import pandas as pd
from LRFutils import logs, progress, color
from src.mltools import sysinfo
from src.mltools.dataset import Dataset
import os


print("\nImporting tensorflow -----------------------------------------------------------\n")
import tensorflow as tf
mae = tf.keras.metrics.MeanAbsoluteError(name="MAE")
from keras.layers import Input, Dense, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Reshape, Conv3DTranspose, Flatten, Concatenate, Dropout
from keras.models import Model
import keras.backend as K
mae = "mae"
print("\nEnd importing tensorflow -------------------------------------------------------\n")

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

def load_file(path:str):
    """
    Load a dataset vector from a file.
    Vector is a dictionary of numpy arrays that represent your data.
    """

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

dataset = Dataset(name="Pres stellar cores", loader=load_file, raw_path=raw_dataset_path, archive_path=dataset_archive)

#==============================================================================
# BUILD MODEL
#==============================================================================

def get_model(dataset):

    sample_x = dataset[0].x

    # Inputs ----------------------------------------------------------------------

    inputs = {
        "Dust map": Input(shape=sample_x["Dust map"].shape, name="Dust map"),
        # "CO cube": Input(shape=sample_x["CO cube"].shape, name="CO cube"),
    }

    # Network ---------------------------------------------------------------------

    x = Flatten()(inputs["Dust map"])
    x = Dense(128, activation='relu')(x)

    x_mass = Dense(32, activation='relu')(x)
    # x_temp = Dense(32, activation='relu')(x)

    # Outputs ---------------------------------------------------------------------

    outputs = {
        "Total mass" : Dense(1, activation='relu', name="Total_mass")(x_mass),
        # "Max temperature" : Dense(1, activation='relu', name="Max_temperature")(x_temp),
    }

    return Model(inputs, outputs)

# Compile, show and train -----------------------------------------------------

# model = get_model(dataset)
# model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

#==============================================================================
# TRAIN MODEL
#==============================================================================