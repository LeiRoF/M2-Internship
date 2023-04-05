print("\nImporting dependencies ---------------------------------------------------------\n")
import numpy as np
import pandas as pd
from LRFutils import logs, archive
import os

from src import mltools

import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Reshape, Conv3DTranspose, Flatten, Concatenate, Dropout
print("\nEnd importing dependencies -----------------------------------------------------\n")

archive_path = archive.new(verbose=True)
print("")

#==============================================================================
# CONFIGURATION
#==============================================================================

valid_frac = 0.2
test_frac  = 0.1
raw_dataset_path = "data/dataset" # path to the raw dataset
dataset_archive = "data/dataset.npz" # path to the dataset archive (avoid redoing data processing)
epochs = 100
batch_size=100
loss = "mean_squared_error"
optimizer = 'adam'
metrics = [
    tf.keras.metrics.MeanAbsoluteError(name="MAE"),
]

#==============================================================================
# LOAD DATASET
#==============================================================================

# Load one file (= vector) ----------------------------------------------------

def load_file(path:str):
    """
    Load a dataset vector from a file.
    Vector is a dictionary of numpy arrays that represent your data.
    """

    data = np.load(path)

    x = {
        "Test":            np.linspace(0, 2*data["mass"], 100),
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

# Load dataset ----------------------------------------------------------------

dataset = mltools.dataset.Dataset(
    name="Pre stellar cores",
    loader=load_file,
    raw_path=raw_dataset_path,
    archive_path=dataset_archive,
    verbose=True
)

#==============================================================================
# BUILD MODEL
#==============================================================================

def get_model(dataset):

    sample_x = dataset[0].x

    # Inputs ----------------------------------------------------------------------

    inputs = [
        Input(shape=sample_x["Test"].shape, name="Test"),
        # Input(shape=sample_x["CO cube"].shape, name="CO cube"),
    ]

    # Network ---------------------------------------------------------------------

    x = Flatten()(inputs[0])

    x_mass = Dense(32, activation='relu')(x)

    # Outputs ---------------------------------------------------------------------

    outputs = [
        Dense(1, activation='sigmoid', name="Total_mass")(x_mass),
    ]

    return mltools.model.Model(inputs, outputs)

# Compile, show and train -----------------------------------------------------

logs.info("Building model...")
model = get_model(dataset)
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
logs.info("Model built. ✅")

model.print()

#==============================================================================
# TRAIN MODEL
#==============================================================================

logs.info("Training model...")
history, trining_time = model.fit(dataset, epochs, batch_size, verbose=True, plot_loss=False)
logs.info("Model trained. ✅")

model.save(archive_path, history=history, training_time=trining_time)

