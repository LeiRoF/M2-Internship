



print("\nImporting dependencies ---------------------------------------------------------\n")

from contextlib import redirect_stdout
import io
from time import time
program_start_time = time()
import numpy as np
from LRFutils import logs, archive, color
import os
import yaml
import json
import matplotlib.pyplot as plt

from src import mltools, physcial_models

os.environ["HDF5_USE_FILE_LOCKINGS"] = "FALSE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
os.environ["http_proxy"] = "http://11.0.0.254:3142/"
os.environ["https_proxy"] = "http://11.0.0.254:3142/"
os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/usr/local/cuda/lib64"

import tensorrt
import tensorflow as tf
# tf.config.experimental.enable_tensor_float_32_execution(enabled=True)
tf.random.set_seed(0)
tf.config.run_functions_eagerly(True)
from keras.layers import Input, Dense, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Reshape, Conv3DTranspose, Flatten, Concatenate, Dropout

print("\nEnd importing dependencies -----------------------------------------------------\n")

archive_path = archive.new(verbose=True)
print("")

#==============================================================================
# CONFIGURATION
#==============================================================================

val_frac = 0.2
test_frac  = 0.1
raw_dataset_path = "data/dataset" # path to the raw dataset
dataset_archive = "data/dataset.npz" # path to the dataset archive (avoid redoing data processing)
epochs = 1000
batch_size=100

#==============================================================================
# LOAD DATASET
#==============================================================================

# Load one file (= vector) ----------------------------------------------------

cpt = 0

def load_file(path:str):
    """
    Load a dataset vector from a file.
    Vector is a dictionary of numpy arrays that represent your data.
    """
    global cpt

    if not cpt%10 == 0:
        return
    cpt += 1

    data = np.load(path)
    vector = mltools.dataset.Vector(

        # x
        dust_wavelenght    = np.array([250.,]), # [um]
        dust_cube          = data["dust_cube"].reshape(*data["dust_cube"].shape, 1), # adding a channel dimension
        dust_map_at_250um  = data["dust_cube"][np.argmin(np.abs(data["dust_freq"]-2.9979e8/250.0e-6)),:,:].reshape(*data["dust_cube"].shape[1:], 1),
        CO_velocity        = data["CO_v"],
        CO_cube            = data["CO_cube"].reshape(*data["CO_cube"].shape, 1), # adding a channel dimension
        N2H_velocity       = data["N2H_v"],
        N2H_cube           = data["N2H_cube"].reshape(*data["N2H_cube"].shape, 1), # adding a channel dimension
        space_range        = data["space_range"],

        # y
        total_mass         = np.array([data["mass"]]),
        max_temperature    = np.array([np.amax(data["dust_temperature"])]),
        plummer_max        = np.array([data["n_H"]]),
        plummer_radius     = np.array([data["r"]]),
        plummer_slope      = np.array([data["p"]]),
        plummer_slope_log  = np.array([np.log(data["p"])]),
        plummer_profile_1D = physcial_models.plummer(
            space_range=data["space_range"],
            max=data["n_H"],
            radius=data["r"],
            slope=data["p"],
        ),
    )

    return vector

# Load dataset ----------------------------------------------------------------

dataset = mltools.dataset.Dataset(
    name="Pre stellar cores",
    loader=load_file,
    raw_path=raw_dataset_path,
    archive_path=dataset_archive,
    val_frac=val_frac,
    test_frac=test_frac,
    verbose=True,
)

#==============================================================================
# BUILD MODEL
#==============================================================================

def get_model(dataset):

    sample = dataset[0]

    # Inputs ----------------------------------------------------------------------

    inputs = {
        "dust_map_at_250um": Input(shape=sample.data["dust_map_at_250um"].shape, name="dust_map_at_250um"),
    }

    # Network ---------------------------------------------------------------------

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs["dust_map_at_250um"])
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x, training=True)
    # Pmax = Dense(128, activation='relu')(x)
    # Prad = Dense(128, activation='relu')(x)
    # Pslope = Dense(128, activation='relu')(x)
    # Pslopelog = Dense(32, activation='relu')(x)
    P1d = Dense(128, activation='relu')(x)

    # Outputs ---------------------------------------------------------------------

    outputs = {
        # "total_mass": Dense(1, activation='sigmoid', name="total_mass")(total_mass),
        # "max_temperature": Dense(1, activation='relu', name="max_temperature")(x),
        # "plummer_max": Dense(1, activation='relu', name="plummer_max")(Pmax),
        # "plummer_radius": Dense(1, activation='relu', name="plummer_radius")(Prad),
        # "plummer_slope": Dense(1, activation='relu', name="plummer_slope")(Pslope),
        # "plummer_slope_log": Dense(1, activation='relu', name="plummer_slope_log")(Pslopelog),
        "plummer_profile_1D": Dense(64, activation='sigmoid', name="plummer_profile_1D")(P1d),
    }

    return mltools.model.Model(inputs, outputs, dataset=dataset, verbose=True)

# Options ---------------------------------------------------------------------

loss="mean_squared_error"
optimizer='RMSprop'
metrics=[
    tf.keras.metrics.MeanAbsoluteError(name="MAE"),
]

# Compile, show and train -----------------------------------------------------

logs.info("Building model...")
model = get_model(dataset)
model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=metrics
)
logs.info("Model built. ✅")

#==============================================================================
# TRAIN MODEL
#==============================================================================

history, trining_time, scores = model.fit(epochs, batch_size, verbose=True, plot_loss=False)

#==============================================================================
# SAVE RESULTS
#==============================================================================

# Save model reference
new_model = model.save_reference("data/model_comparison", archive_path)

# Convert metrics to strings
for i, metric in enumerate(metrics):
    if not isinstance(metric, str):
        metrics[i] = metric.name

# Save model details
model.save(
    archive_path,
    history=history,
    training_time=trining_time,
    val_frac=val_frac,
    test_frac=test_frac,
    epochs=epochs,
    batch_size=batch_size,
    loss=loss,
    optimizer=optimizer,
    metrics=metrics,
    model_id=new_model,
    dataset=str(model.dataset).split("\n"),
    scores=scores,
)

# End of program
spent_time = time() - program_start_time
logs.info(f"End of program. ✅ Took {int(spent_time//60)} minutes and {spent_time%60:.2f} seconds \n -> Results dans be found in {archive_path} folder.")

#==============================================================================
# PREDICTION
#==============================================================================

print("\n\nPredictions --------------------------------------------------------------------\n\n")

p = model.predict(display=True, N=5, save_as=f"{archive_path}/predictions.npz")
