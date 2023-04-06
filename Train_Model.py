print("\nImporting dependencies ---------------------------------------------------------\n")
import numpy as np
import pandas as pd
from LRFutils import logs, archive
import os

from src import mltools

os.environ["HDF5_USE_FILE_LOCKINGS"] = "FALSE"
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
epochs = 1000
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
        "Test":            np.linspace(0, 2*data["mass"], 3),
        "Dust_wavelenght": np.array([250.,]), # [um]
        "Dust_map" :       data["dust_image"].reshape(*data["dust_image"].shape, 1), # adding a channel dimension
        "CO_velocity" :    data["CO_v"],
        "CO_cube" :        data["CO_cube"].reshape(*data["CO_cube"].shape, 1), # adding a channel dimension
        "N2H+_velocity" :  data["N2H_v"],
        "N2H_cube" :       data["N2H_cube"].reshape(*data["N2H_cube"].shape, 1), # adding a channel dimension
    }

    y = {
        "Total_mass" :      np.array([data["mass"]]),
        "Max_temperature" : np.array([np.amax(data["dust_temperature"])]),
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

    inputs = {
        "Test": Input(shape=sample_x["Test"].shape, name="Test"),
    }

    # Network ---------------------------------------------------------------------

    x = Flatten()(inputs["Test"])

    x_mass = Dense(32, activation='relu')(x)

    # Outputs ---------------------------------------------------------------------

    outputs = {
        "Total_mass": Dense(1, activation='sigmoid', name="Total_mass")(x_mass),
    }

    return mltools.model.Model(inputs, outputs, dataset=dataset, verbose=True)

# Compile, show and train -----------------------------------------------------

logs.info("Building model...")
model = get_model(dataset)
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
logs.info("Model built. ✅")

#==============================================================================
# TRAIN MODEL
#==============================================================================

logs.info("Training model...")
history, trining_time = model.fit(epochs, batch_size, verbose=True, plot_loss=False)
logs.info("Model trained. ✅")

model.save(archive_path, history=history, training_time=trining_time)


logs.info(f"Done. ✅\n -> Results dans be found in {archive_path} folder.")
