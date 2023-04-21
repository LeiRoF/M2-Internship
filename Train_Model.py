print("\nImporting dependencies ---------------------------------------------------------\n")
from time import time
program_start_time = time()
import numpy as np
from LRFutils import logs, archive
import os
import yaml
import json
import matplotlib.pyplot as plt

from src import mltools

os.environ["HDF5_USE_FILE_LOCKINGS"] = "FALSE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
tf.random.set_seed(0)
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
epochs = 100
batch_size=100

#==============================================================================
# LOAD DATASET
#==============================================================================

# Load one file (= vector) ----------------------------------------------------

cpt = -1

mass = []

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

    def normalized_plummer(n_H:float, M:float, d:float, r:float, p:float) -> float:
        return 3 * M /(4 * np.pi * r**3) * (1 + np.abs(d)**p / r**p)**(-5/2)

    vector = mltools.dataset.Vector(

        # x
        Dust_wavelenght    = np.array([250.,]), # [um]
        Dust_cube          = data["dust_cube"].reshape(*data["dust_cube"].shape, 1), # adding a channel dimension
        Dust_map_at_250um  = data["dust_cube"][data["dust_cube"].shape[0]//2,:,:].reshape(*data["dust_cube"].shape[1:], 1),
        # Dust_map_at_250um  = data["dust_cube"][np.argmin(np.abs(data["dust_freq"]-2.9979e8/250.0e-6)),:,:].reshape(*data["dust_cube"].shape[:1], 1)
        CO_velocity        = data["CO_v"],
        CO_cube            = data["CO_cube"].reshape(*data["CO_cube"].shape, 1), # adding a channel dimension
        N2H_velocity       = data["N2H_v"],
        N2H_cube           = data["N2H_cube"].reshape(*data["N2H_cube"].shape, 1), # adding a channel dimension

        # y
        Total_mass         = np.array([data["mass"]]),
        Max_temperature    = np.array([np.amax(data["dust_temperature"])]),
        Plummer_max        = np.array([data["n_H"]]),
        Plummer_radius     = np.array([data["r"]]),
        Plummer_slope      = np.array([data["p"]]),
        Plummer_profile_1D = normalized_plummer(
            n_H = data["n_H"],
            M = 1,
            d = np.linspace(-25, 25, 3, endpoint=True),
            r = data["r"],
            p = data["p"],
        ),

    )

    mass.append(data["mass"])

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

# plt.figure()
# plt.hist(mass, bins=100)
# plt.savefig(f"Total_mass_before.png")

# plt.figure()
# plt.hist(dataset.denormalize("Total_mass", dataset.data["Total_mass"]), bins=100)
# plt.savefig(f"Total_mass_after.png")

# dataset.print_few_vectors()

#==============================================================================
# BUILD MODEL
#==============================================================================

def get_model(dataset):

    sample = dataset[0]

    # Inputs ----------------------------------------------------------------------

    inputs = {
        "Dust_map_at_250um": Input(shape=sample.data["Dust_map_at_250um"].shape, name="Dust_map_at_250um"),
    }

    # Network ---------------------------------------------------------------------

    x = Flatten()(inputs["Dust_map_at_250um"])
    # x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x, training=True)
    # Plummer_max = Dense(32, activation='relu')(x)
    # Plummer_radius = Dense(32, activation='relu')(x)
    # Plummer_slope = Dense(32, activation='relu')(x)

    # Outputs ---------------------------------------------------------------------

    outputs = {
        "Total_mass": Dense(1, activation='sigmoid', name="Total_mass")(x),
        # "Max_temperature": Dense(1, activation='relu', name="Max_temperature")(x),
        # "Plummer_max": Dense(1, activation='relu', name="Plummer_max")(Plummer_max),
        # "Plummer_radius": Dense(1, activation='relu', name="Plummer_radius")(Plummer_radius),
        # "Plummer_slope": Dense(1, activation='relu', name="Plummer_slope")(Plummer_slope),
        "Plummer_profile_1D": Dense(3, activation='sigmoid', name="Plummer_profile_1D")(x),
    }

    return mltools.model.Model(inputs, outputs, dataset=dataset, verbose=True)

# Options ---------------------------------------------------------------------

loss="mean_squared_error"
optimizer='SGD'
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

model.predict(display=True, N=5, save_as=f"{archive_path}/predictions.npz")