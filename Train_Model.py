#==============================================================================
# IMPORT DEPENDENCIES
#==============================================================================

print("\nImporting dependencies ---------------------------------------------------------\n")

from contextlib import redirect_stdout
import io
from time import time

from src import physical_models
program_start_time = time()
import numpy as np
from LRFutils import logs, archive, color, progress
import os
import yaml
import json
import matplotlib.pyplot as plt

from src import mltools

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
# tf.config.run_functions_eagerly(True)
from keras.layers import Input, Dense, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Reshape, Conv3DTranspose, Flatten, Concatenate, Dropout

print("\nEnd importing dependencies -----------------------------------------------------\n")

archive_path = archive.new(verbose=True)
print("")

#==============================================================================
# LOAD DATASET
#==============================================================================

# Load dataset ----------------------------------------------------------------

cpt = 0
dataset_path = "data/dataset_old"

logs.info("Loading dataset...")
bar = progress.Bar(max=len(os.listdir(dataset_path)))

vectors = []
for file in os.listdir(dataset_path):

    cpt += 1
    if not cpt%10 == 0:
        continue

    bar(cpt, prefix=mltools.sysinfo.get())
    
    file_path = os.path.join(dataset_path, file)

    data = np.load(file_path)

    # dust_map = data["dust_cube"][np.argmin(np.abs(data["dust_freq"]-2.9979e8/250.0e-6)),:,:]

    # if not os.path.isdir("test"):
    #     os.mkdir("test")
    # plt.figure()
    # plt.imshow(dust_map)
    # plt.title(f"nH={data['n_H']}, r={data['r']}, p={data['p']}")
    # plt.savefig(f"test/{cpt}.png")

    vectors.append(mltools.vector.Vector(
        
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
        plummer_profile_1D = physical_models.plummer(
            space_range=data["space_range"],
            max=data["n_H"],
            radius=data["r"],
            slope=data["p"],
        ),
    ))

dataset = mltools.dataset.Dataset(vectors)
print(dataset)

logs.info("Dataset loaded. ✅")

# Process dataset -------------------------------------------------------------

val_frac = 0.2
test_frac  = 0.1
# dataset = dataset.normalize()
train_dataset, val_dataset, test_dataset = dataset.split(val_frac, test_frac)

#==============================================================================
# BUILD MODEL
#==============================================================================

# Design model ----------------------------------------------------------------

# Inputs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

input_fields = [
    "dust_map_at_250um",
]

inputs = {}
normalized_inputs = {}
for label in input_fields:
    inputs[label] = Input(shape=dataset.shapes[label], name=label)
    normalized_inputs[label] = tf.keras.layers.Normalization(axis=None, mean=dataset[label].mean(), variance=dataset[label].std(), invert=False)(inputs[label])

# Network ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	

x = Conv2D(16, (5, 5), activation='linear', padding='same')(normalized_inputs["dust_map_at_250um"])
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
x = Dense(128, activation='linear')(x)
x = Dropout(0.3)(x, training=True)
# Pmax = Dense(128, activation='linear')(x)
Prad = Dense(128, activation='linear')(x)
# Pslope = Dense(128, activation='linear')(x)
# Pslopelog = Dense(32, activation='linear')(x)
# P1d = Dense(128, activation='linear')(x)

# Outputs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

normalized_outputs = {
    # "total_mass": Dense(1, activation='linear')(total_mass),
    # "max_temperature": Dense(1, activation='linear')(x),
    # "plummer_max": Dense(1, activation='linear')(Pmax),
    "plummer_radius": Dense(1, activation='linear')(Prad),
    # "plummer_slope": Dense(1, activation='linear')(Pslope),
    # "plummer_slope_log": Dense(1, activation='linear')(Pslopelog),
    # "plummer_profile_1D": Dense(64, activation='linear')(P1d),
}

outputs = {}
for label, value in normalized_outputs.items():
    outputs[label] = tf.keras.layers.Normalization(axis=None, mean=dataset[label].mean(), variance=dataset[label].std(), invert=True, name=label)(value)

model = mltools.model.Model(inputs, outputs)


# Compile, show and train -----------------------------------------------------

logs.info("Building model...")

loss="mean_squared_error"
optimizer="adam"
metrics=[tf.keras.metrics.MeanAbsoluteError(name="MAE"),]

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
model.print()
print(model.input_names)

logs.info("Model built. ✅")

#==============================================================================
# TRAIN MODEL
#==============================================================================

epochs = 1000
batch_size=100

history, trining_time = model.fit(train_dataset, epochs, batch_size, val=val_dataset, verbose=True, plot_loss=False)

#==============================================================================
# PREDICTION
#==============================================================================

print("\n\nPredictions --------------------------------------------------------------------\n\n")

p = model.predict(test_dataset, display=True, N=100, save_as=f"{archive_path}/inference.npz")

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
    dataset_summary=str(dataset),
    # dataset_summary=str(dataset.denormalize()),
    # dataset_means=dataset.mean(),
    # dataset_stds=dataset.std(),
    # dataset_mins=dataset.min(),
    # dataset_maxs=dataset.max(),
    dataset_shapes=dataset.shapes,
    dataset_size=len(dataset),
)

np.savez_compressed(f"{archive_path}/test_set.npz", test_set=dict(test_dataset.filter(model.input_names + model.output_names)))

# End of program
spent_time = time() - program_start_time
logs.info(f"End of program. ✅ Took {int(spent_time//60)} minutes and {spent_time%60:.2f} seconds \n -> Results dans be found in {archive_path} folder.")
