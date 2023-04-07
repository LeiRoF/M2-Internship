print("\nImporting dependencies ---------------------------------------------------------\n")
from time import time
program_start_time = time()
import numpy as np
import pandas as pd
from LRFutils import logs, archive
import os
import yaml

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

val_frac = 0.2
test_frac  = 0.1
raw_dataset_path = "data/dataset" # path to the raw dataset
dataset_archive = "data/dataset.npz" # path to the dataset archive (avoid redoing data processing)
epochs = 1000
batch_size=10
loss = "mean_squared_error"
optimizer = 'adam'
metrics = [
    tf.keras.metrics.MeanAbsoluteError(name="Total_mass_MAE"),
    # tf.keras.metrics.MeanAbsoluteError(name="Max_temperature_MAE"),
]

#==============================================================================
# LOAD DATASET
#==============================================================================

# Load one file (= vector) ----------------------------------------------------

cpt = -1

def load_file(path:str):
    """
    Load a dataset vector from a file.
    Vector is a dictionary of numpy arrays that represent your data.
    """

    global cpt

    cpt += 1
    if not cpt%10 == 0:
        return

    data = np.load(path)

    x = {
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
    val_frac=val_frac,
    test_frac=test_frac,
    verbose=True,
)

#==============================================================================
# BUILD MODEL
#==============================================================================

def get_model(dataset):

    sample_x = dataset[0].x

    # Inputs ----------------------------------------------------------------------

    inputs = {
        "Dust_map": Input(shape=sample_x["Dust_map"].shape, name="Dust_map"),
    }

    # Network ---------------------------------------------------------------------

    x = Flatten()(inputs["Dust_map"])

    x_mass = Dense(32, activation='relu')(x)

    # Outputs ---------------------------------------------------------------------

    outputs = {
        "Total_mass": Dense(1, activation='sigmoid', name="Total_mass")(x_mass),
        # "Max_temperature": Dense(1, activation='relu', name="Max_temperature")(x_mass),
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
)

# End of program
spent_time = time() - program_start_time
logs.info(f"End of program. ✅ Took {int(spent_time//60)} minutes and {spent_time%60:.2f} seconds \n -> Results dans be found in {archive_path} folder.")

#==============================================================================
# PREDICTION
#==============================================================================

print("\n\nPREDICTION TIME!\n\n")

y_prediction = model.predict(model.dataset.test.x)

for key, value in y_prediction.items():
    print(f"   {key} :")
    for i, prediction in enumerate(value):
        p = prediction.flatten()[0] * model.dataset.ystds[key] + model.dataset.ymeans[key]
        y = dataset.test.y[key][i].flatten()[0] * dataset.ystds[key] + dataset.ymeans[key]
        r = dataset.ystds[key]
        print(f"      {i} : Predicted: {p:.2e}, Expected: {y:.2e}, Error: {(p-y):.2e} ({np.abs(p-y)/r:.2f} σ)")