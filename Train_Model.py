#==============================================================================
# INITIALIZATION
#==============================================================================



# Dependencies ----------------------------------------------------------------

import numpy as np
import tensorflow as tf
from LRFutils import archive, progress
import os
from multiprocess import Pool, cpu_count
import psutil

# Config ----------------------------------------------------------------------

valid_frac = 0.2
test_frac  = 0.1
dataset_path = "/scratch/vforiel/dataset"

# Global variables ------------------------------------------------------------

archive_path = archive.new()

try:
    ncpu = cpu_count()
except:
    with open(os.getenv("OAR_NODEFILE"), 'r') as f:
        ncpu = len(f.readlines())

# Useful functions ------------------------------------------------------------

def system_info():
    return f"CPU: {psutil.cpu_percent()}%"\
        + f", RAM: {psutil.virtual_memory().percent}%"\
        + f" ({psutil.virtual_memory().used/1024**3:.2f}GB"\
        + f"/ {psutil.virtual_memory().total/1024**3:.2f}GB)"


#==============================================================================
# LOAD DATA
#==============================================================================



"""
For one vector, we expect something like this:

**Input $x_i$:**
- I[x,y,f] : data cube of intensity for a given pixel (x,y) and frequency f

**Outputs $y_i$:**
- Vx[x,y,z] : data cube of velocity in x direction for a given coordinate in space (x,y,z)
- Vy[x,y,z] : data cube of velocity in y direction for a given coordinate in space (x,y,z)
- Vz[x,y,z] : data cube of velocity in z direction for a given coordinate in space (x,y,z)
- $\rho$[x,y,z] : data cube of density for a given coordinate in space (x,y,z)

In practice, there is lot of vectors, so $x = [x_1, x_2, ..., x_N]$ and $y = [y_1, y_2, ..., y_N]$ where $N$ is the number of vectors.

To simplify the neural network and potentially increase it's accuracy, we will not design a network that predicts all of the outputs at once. Instead, we will design 4 networks that will predict one output. This means that we will have 4 networks, one for each output. So $y_i$ will alternatively contain only one of the elements listed above.

> **YOUR JOB**: In the following cell, write the code that load the data as specified above.
"""

# Read one file ---------------------------------------------------------------

def load_file(file):
    data = np.load(f"{dataset_path}/{file}")
    return data

# Load data -------------------------------------------------------------------

def load_data() -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Do what you want int this function, as long as it returns the following:
    - list[3D-ndarray] : input vectors
    - list[3D-ndarray] : output vectors
    """

    max_files = 1000

    files = os.listdir(dataset_path)

    nb_files = min(len(files), max_files)

    with np.load(f"{dataset_path}/{files[0]}") as data:
        x = np.empty((nb_files, *data["observation"].shape))
        y = np.empty((nb_files, *data["vz"].shape))

    bar = progress.Bar(nb_files, "Loading data")
    for i, file in enumerate(files):
        if i >= max_files:
            break
        
        bar(i, prefix=system_info())

        data = np.load(f"{dataset_path}/{file}")
        x[i] = data["observation"]
        y[i] = data["vz"]
        
    bar(nb_files)

    print("x shape: ", np.array(x).shape)
    
    return x, y



#=============================================================================
# Post data treatment
#=============================================================================



# This part only consist to check the data consistency,
# normalize it and split the dataset.

# Check data consistency -----------------------------------------------------

x, y = load_data()
assert len(x) == len(y), f"x and y must have the same length, found {len(x)}, {len(y)}"
nb_vectors = len(x)

# Normalizing data ------------------------------------------------------------

x /= np.max(np.abs(x))
y /= np.max(np.abs(y))

# Splitting datasets ----------------------------------------------------------

train_frac = 1 - valid_frac - test_frac

train_x = x[:int(nb_vectors * train_frac)]
train_y = y[:int(nb_vectors * train_frac)]

valid_x = x[int(nb_vectors * train_frac):int(nb_vectors * (train_frac + valid_frac))]
valid_y = y[int(nb_vectors * train_frac):int(nb_vectors * (train_frac + valid_frac))]

test_x = x[int(nb_vectors * (train_frac + valid_frac)):]
test_y = y[int(nb_vectors * (train_frac + valid_frac)):]



#==============================================================================
# Model definition
#==============================================================================



# Build the 3D CNN model ------------------------------------------------------

def get_model(input_shape, output_shape):
    from keras.layers import Input, Dense, Conv2D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Reshape, Conv3DTranspose, Flatten
    from keras.models import Model

    # Définir la forme de l'image d'entrée
    input_img = Input(shape=(64, 64, 100))

    # Encoder
    x = Conv2D(32, (15, 15), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)

    # Decoder
    x = Dense(32 * 4 * 4 * 4, activation='relu')(x)
    x = Reshape((4, 4, 4, 32))(x)
    x = Conv3DTranspose(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((4, 4, 4))(x)
    x = Conv3DTranspose(1, (5, 5, 5), activation='relu', padding='same')(x)
    decoded = UpSampling3D((4, 4, 4))(x)

    # Modèle d'auto-encodeur
    autoencoder = Model(input_img, decoded)

    return autoencoder

# Compile and get summary -----------------------------------------------------

model = get_model(x[0].shape, y[0].shape)
# model.compile(optimizer='adam', loss='binary_crossentropy')
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()

choice = input("Continue ? [Y/n]")

if choice.lower() not in ["", "y", "yes"]:
    exit()

# Training model --------------------------------------------------------------

print("++++++")

print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)

print("++++++")

model.fit(train_x, train_y, epochs=10, batch_size=32, validation_data=(valid_x, valid_y))
model.save(f'{archive_path}/model0.h5')

# Evaluating model -----------------------------------------------------------

score = model.evaluate(test_x, test_y, verbose=0)
print("Score:", score)

with open(f'{archive_path}/scores.txt', 'w') as f:
    f.write(f'Score:    \t{score}\n')

# Prediction -----------------------------------------------------------------

x_prediction = [x[0,...]]
x_prediction = np.expand_dims(x_prediction, axis=-1)
print(x_prediction.shape)

y_prediction = model.predict(x_prediction)
print(y_prediction.shape)

np.savez_compressed(f'{archive_path}/prediction.npz', x=x_prediction, y=y_prediction)