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

# Global variables ------------------------------------------------------------

archive_path = archive.new()

try:
    ncpu = cpu_count()
except:
    with open(os.getenv("OAR_NODEFILE"), 'r') as f:
        ncpu = len(f.readlines())



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
    data = np.load("dataset/" + file)
    return data

# Load data -------------------------------------------------------------------

def load_data() -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Do what you want int this function, as long as it returns the following:
    - list[3D-ndarray] : input vectors
    - list[3D-ndarray] : output vectors
    """
    x = []
    y0 = []
    y1 = []
    y2 = []
    y3 = []

    window_size = 5

    width = (window_size-1)//2

    for file in os.listdir("dataset/")[0:2]:
        data = np.load("dataset/" + file)
        crop = np.arange(width, data["observation"].shape[1]-width)
        
        bar = progress.Bar(len(crop)**2, prefix=f"{file}, {len(x)}")
        for n, i in enumerate(crop):
            for m, j in enumerate(crop):
                
                prefix = f"CPU: {psutil.cpu_percent()}%, RAM: {psutil.virtual_memory().percent}% ({psutil.virtual_memory().used/1024**3:.2f}GB / {psutil.virtual_memory().total/1024**3:.2f}GB)"
                
                bar(n*len(crop)+m, prefix=f"{file} | {prefix}")
                # print("------")
                # print(file,i,j)
                x.append(data["observation"][i-width:i+width+1, j-width:j+width+1, :])
                # print("x",x[-1].shape)
                y0.append(data["cloud"][i, j, :])
                # print("y0",y0[-1].shape)
                y1.append(data["vx"][i, j, :])
                # print("y1",y1[-1].shape)
                y2.append(data["vy"][i, j, :])
                # print("y2",y2[-1].shape)
                y3.append(data["vz"][i, j, :])
                # print("y3",y3[-1].shape)
    
    bar(len(crop))

    print("x shape: ", np.array(x).shape)
    
    return x, y0, y1, y2, y3



#=============================================================================
# Post data treatment
#=============================================================================



"""
This part only consist to check the data consistency, normalize it and split the dataset.
"""

# Check data consistency -----------------------------------------------------

x, y0, y1, y2, y3 = load_data()
assert len(x) == len(y0) == len(y1) == len(y2) == len(y3), f"x and y must have the same length, found {len(x)}, {len(y0)}, {len(y1)}, {len(y2)}, {len(y3)}"
x = np.array(x)
y0 = np.array(y0)
y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)

# Getting dimensions ---------------------------------------------------------

nb_vectors = len(x)

len_x = x.shape[3]
len_y0 = y0.shape[3]
len_y1 = y1.shape[3]
len_y2 = y2.shape[3]
len_y3 = y3.shape[3]

# Normalizing data ------------------------------------------------------------

x /= np.max(np.abs(x))
y0 /= np.max(np.abs(y0))
y1 /= np.max(np.abs(y1))
y2 /= np.max(np.abs(y2))
y3 /= np.max(np.abs(y3))

# Splitting datasets ----------------------------------------------------------

train_frac = 1 - valid_frac - test_frac

train_x = x[:int(nb_vectors * train_frac)]
train_y0 = y0[:int(nb_vectors * train_frac)]
train_y1 = y1[:int(nb_vectors * train_frac)]
train_y2 = y2[:int(nb_vectors * train_frac)]
train_y3 = y3[:int(nb_vectors * train_frac)]
train_y = [train_y0, train_y1, train_y2, train_y3]

valid_x = x[int(nb_vectors * train_frac):int(nb_vectors * (train_frac + valid_frac))]
valid_y0 = y0[int(nb_vectors * train_frac):int(nb_vectors * (train_frac + valid_frac))]
valid_y1 = y1[int(nb_vectors * train_frac):int(nb_vectors * (train_frac + valid_frac))]
valid_y2 = y2[int(nb_vectors * train_frac):int(nb_vectors * (train_frac + valid_frac))]
valid_y3 = y3[int(nb_vectors * train_frac):int(nb_vectors * (train_frac + valid_frac))]
valid_y = [valid_y0, valid_y1, valid_y2, valid_y3]

test_x = x[int(nb_vectors * (train_frac + valid_frac)):]
test_y0 = y0[int(nb_vectors * (train_frac + valid_frac)):]
test_y1 = y1[int(nb_vectors * (train_frac + valid_frac)):]
test_y2 = y2[int(nb_vectors * (train_frac + valid_frac)):]
test_y3 = y3[int(nb_vectors * (train_frac + valid_frac)):]
test_y = [test_y0, test_y1, test_y2, test_y3]

#==============================================================================
# Model definition
#==============================================================================

# Build the 3D CNN model ------------------------------------------------------

def get_model(input_shape, output_shape):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv3D(32, kernel_size=(15, 15, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
    # > Utile ?
    # model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(tf.keras.layers.Conv3D(64, kernel_size=(5, 5, 3), activation='relu', kernel_initializer='he_uniform'))
    # model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))
    # <
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(np.prod(output_shape), activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(np.prod(output_shape), activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(np.prod(output_shape), activation='relu', kernel_initializer='he_uniform'))
    # > Utile ?
    # model.add(tf.keras.layers.Dense(np.prod(output_shape), activation='softmax'))
    # <
    model.add(tf.keras.layers.Reshape(output_shape))

    return model

# # Training models

# %%
# Training model for y0

model0 = get_model(input_shape, output0_shape)
model0.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model0.fit(train_x, train_y0, epochs=10, batch_size=32, validation_data=(valid_x, valid_y0))
model0.save(f'{archive_path}/model0.h5')

# %%
# Training model for y1

model1 = get_model(input_shape, output0_shape)
model1.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model1.fit(train_x, train_y1, epochs=10, batch_size=32, validation_data=(valid_x, valid_y1))
model1.save(f'{archive_path}/model1.h5')

# %%
# Training model for y2

model2 = get_model(input_shape, output0_shape)
model2.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model2.fit(train_x, train_y2, epochs=10, batch_size=32, validation_data=(valid_x, valid_y2))
model2.save(f'{archive_path}/model2.h5')

# %%
# Training model for y3

model3 = get_model(input_shape, output0_shape)
model3.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model3.fit(train_x, train_y3, epochs=10, batch_size=32, validation_data=(valid_x, valid_y3))
model3.save(f'{archive_path}/model3.h5')

# %% [markdown]
# ---
# 
# # Evaluating models

# %%
# Evaluate model for y0

score0 = model0.evaluate(test_x, test_y0, verbose=0)
print('Test loss:', score0[0])
print('Test accuracy:', score0[1])

# %%
# Evaluate model for y1

score1 = model1.evaluate(test_x, test_y1, verbose=0)
print('Test loss:', score1[0])
print('Test accuracy:', score1[1])

# %%
# Evaluate model for y2

score2 = model2.evaluate(test_x, test_y2, verbose=0)
print('Test loss:', score2[0])
print('Test accuracy:', score2[1])

# %%
# Evaluate model for y3

score3 = model3.evaluate(test_x, test_y3, verbose=0)
print('Test loss:', score3[0])
print('Test accuracy:', score3[1])

# %%
with open(f'{archive_path}/scores.txt', 'w') as f:
    f.write('\t\t\t\tModel 0\tModel 1\tModel 2\tModel 3\n')
    f.write(f'Test loss:    \t{round(score0[0],3)}\t{round(score1[0],3)}\t{round(score2[0],3)}\t{round(score3[0],3)}\n')
    f.write(f'Test accuracy:\t{round(score0[1],3)}\t{round(score1[1],3)}\t{round(score2[1],3)}\t{round(score3[1],3)}\n')

# %% [markdown]
# ---
# 
# # Prediction

# %%
print(x.shape)
x_prediction = [x[0,...]]
x_prediction = np.expand_dims(x_prediction, axis=-1)
print(x_prediction.shape)

y0_prediction = model0.predict(x_prediction)
print(y0_prediction.shape)

# y1_prediction = model1.predict([x_prediction])
# print(y1_prediction.shape)

# y2_prediction = model2.predict([x_prediction])
# print(y2_prediction.shape)

# y3_prediction = model3.predict([x_prediction])
# print(y3_prediction.shape)


