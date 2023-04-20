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
total_mass_mae = tf.keras.metrics.MeanAbsoluteError(name="Total_mass_MAE")
print("\nEnd importing dependencies -----------------------------------------------------\n")

archive_path = archive.new(verbose=True)
print("")

#==============================================================================
# CONFIGURATION
#==============================================================================

val_frac = 0.2
test_frac  = 0.1
raw_dataset_path = "data/dataset_old" # path to the raw dataset
dataset_archive = "data/dataset.npz" # path to the dataset archive (avoid redoing data processing)
epochs = 100000
batch_size=100
loss = "mean_squared_error"
optimizer = 'SGD'
metrics = [
    total_mass_mae,
    # tf.keras.metrics.MeanAbsoluteError(name="Max_temperature_MAE"),
]

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
    if not cpt%1 == 0:
        return

    data = np.load(path)

    vector = mltools.dataset.Vector(

        # x
        Dust_wavelenght = np.array([250.,]), # [um]
        Dust_map =        data["dust_image"].reshape(*data["dust_image"].shape, 1), # adding a channel dimension
        CO_velocity =     data["CO_v"],
        CO_cube =         data["CO_cube"].reshape(*data["CO_cube"].shape, 1), # adding a channel dimension
        N2H_velocity =    data["N2H_v"],
        N2H_cube =        data["N2H_cube"].reshape(*data["N2H_cube"].shape, 1), # adding a channel dimension

        # y
        Total_mass =      np.array([data["mass"]]),
        Max_temperature = np.array([np.amax(data["dust_temperature"])]),
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
        "Dust_map": Input(shape=sample.data["Dust_map"].shape, name="Dust_map"),
    }

    # Network ---------------------------------------------------------------------

    x = Conv2D(16, (5, 5), activation='relu', padding='same')(inputs["Dust_map"])
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x, training=True)
    x = Dense(128, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Outputs ---------------------------------------------------------------------

    outputs = {
        "Total_mass": Dense(1, activation='sigmoid', name="Total_mass")(x),
        # "Max_temperature": Dense(1, activation='relu', name="Max_temperature")(x),
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

print(model.dataset.test.means)
print(model.dataset.test.stds)

print("\n\nPredictions --------------------------------------------------------------------\n\n")

predictions = {}
expectations = model.dataset.test.y.data
for key in expectations.keys():
    expectations[key] = model.dataset.denormalize(key, expectations[key])
    
for i in range(1000):
    prediction, _ = model.predict(model.dataset.test.x.data, display=False)

    for key in prediction:
        if not key in predictions:
            predictions[key] = []
        predictions[key].append(prediction[key])                               
        
np.savez_compressed(f"{archive_path}/predictions.npz", predictions=predictions, expectations=expectations)