#==============================================================================
# INITIALIZATION
#==============================================================================



# Environment -----------------------------------------------------------------

# run "nvidia-smi" to check the GPU properties

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Dependencies ----------------------------------------------------------------

import numpy as np
import tensorflow as tf
from LRFutils import archive, progress
from multiprocess import Pool, cpu_count
import psutil
import pandas as pd
import matplotlib.pyplot as plt


print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Configuration ---------------------------------------------------------------

valid_frac = 0.2
test_frac  = 0.1
dataset_path = "data/dataset"

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



# Data properties -------------------------------------------------------------

def data_labels(x,y):
    """Take a vector of a dataset and return it's properties"""

    x_labels = [
        # "Dust Obs. Wavelenght [um]",
        "Dust Map",
        # "CO Velocity",
        # "CO Cube",
        # "N2H+ Velocity",
        # "N2H Cube"
    ]
    
    y_labels = ["Mass"]

    return x_labels, y_labels

# Read one file ---------------------------------------------------------------

def load_file(file):

    data = np.load(file)

    x = [
        # np.array(250), # dust observation frequency [um}
        data["dust_image"].reshape(*data["dust_image"].shape, 1), # adding a channel dimension
        # data["CO_v"],
        # data["CO_cube"].reshape(*data["CO_cube"].shape, 1), # adding a channel dimension
        # data["N2H_v"],
        # data["N2H_cube"].reshape(*data["N2H_cube"].shape, 1) # adding a channel dimension
    ]
    
    y = [np.array(data["mass"]),]

    return x, y

# Load data -------------------------------------------------------------------

def load_data() -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Do what you want int this function, as long as it returns the following:
    - list[3D-ndarray] : input vectors
    - list[3D-ndarray] : output vectors
    """

    # Limit of the number of vectors to load
    max_files = 1000
    files = os.listdir(dataset_path)
    nb_vectors = min(len(files), max_files)

    # Load data
    x = []
    y = []
    print("\nLoading data...")
    bar = progress.Bar(nb_vectors)
    for i, file in enumerate(files):
        if i >= nb_vectors:
            break
        bar(i, prefix=system_info())
        
        new_x, new_y = load_file(f"{dataset_path}/{file}")
        x.append(new_x)
        y.append(new_y)
        
    bar(nb_vectors)    
    return x, y

x, y = load_data()
x_labels, y_labels = data_labels(x[0], y[0])
nb_vectors = len(x)



#==============================================================================
# POST PROCESSING
#==============================================================================



# Normalisation ---------------------------------------------------------------

x_maxs = []
for element in x[0]:
    x_maxs.append(element.ravel()[0])

y_maxs = []
for element in y[0]:
    y_maxs.append(element.ravel()[0])

for vector in x:
    for i in range(len(vector)):
        if (value := np.max(np.abs(vector[i]))) > x_maxs[i]:
            x_maxs[i] = value

for i in range(len(x)):
    for j in range(len(vector)):
        x[i][j] /= x_maxs[j]

for vector in y:
    for i in range(len(vector)):
        if (value := np.max(np.abs(vector[i]))) > y_maxs[i]:
            y_maxs[i] = value

for i in range(len(y)):
    for j in range(len(vector)):
        y[i][j] /= y_maxs[j]

# Splitting datasets ----------------------------------------------------------

# Train
train_frac = 1 - valid_frac - test_frac

train_x = x[:int(nb_vectors * train_frac)]
train_y = y[:int(nb_vectors * train_frac)]

# Validation

valid_x = x[int(nb_vectors * train_frac):int(nb_vectors * (train_frac + valid_frac))]
valid_y = y[int(nb_vectors * train_frac):int(nb_vectors * (train_frac + valid_frac))]

# Test

test_x = x[int(nb_vectors * (train_frac + valid_frac)):]
test_y = y[int(nb_vectors * (train_frac + valid_frac)):]

# Convert to numpy arrays
x = np.array([i[0] for i in x])
train_x = np.array([i[0] for i in train_x])
valid_x = np.array([i[0] for i in valid_x])
test_x = np.array([i[0] for i in test_x])
y = np.array([i[0] for i in y])
train_y = np.array([i[0] for i in train_y])
valid_y = np.array([i[0] for i in valid_y])
test_y = np.array([i[0] for i in test_y])



#==============================================================================
# MODEL DEFINITION
#==============================================================================



# Build model -----------------------------------------------------------------

def get_model(input_shape):
    from keras.layers import Input, Dense, Conv2D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Reshape, Conv3DTranspose, Flatten
    from keras.models import Model

    # Définir la forme de l'image d'entrée
    input = Input(shape=input_shape)

    # Encoder
    # x = Conv2D(8, (5, 5), activation='relu', padding='same')(input)
    # x = MaxPooling2D((4, 4), padding='same')(x)
    # x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    # x = MaxPooling2D((4, 4), padding='same')(x)
    # x = Flatten()(x)
    # x = Dense(1024, activation='relu')(x)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='relu')(x)

    # Modèle d'auto-encodeur
    model = Model(input, output)

    return model

# Compile model and get summary -----------------------------------------------

model = get_model(x[0].shape)

def tf_pearson(y_true, y_pred):
    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]

model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['mean_squared_error'])

# Store and print model summary
stringlist = []
model.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)
print(short_model_summary)

# Human validation of the model -----------------------------------------------

# choice = input("Continue ? [Y/n]")

# if choice.lower() not in ["", "y", "yes"]:
#     exit()

# Training model --------------------------------------------------------------

epochs = 10000

print("\nTraining model...")
bar = progress.Bar(epochs)
bar(0)
stage=0

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global bar
        bar(epoch, prefix = f"MSE: {logs['mean_squared_error']:.2e}")


history = model.fit(train_x, train_y, epochs=10000, batch_size=50, validation_data=(valid_x, valid_y), verbose=0, callbacks=[CustomCallback()])
model.save(f'{archive_path}/model0.h5')
bar(epochs)

plt.plot(history.history['loss'], alpha=0.5, label='Train Loss')
plt.plot(history.history['val_loss'], alpha=0.5, label='Val Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(f'{archive_path}/loss.png')

plt.xscale('log')
plt.yscale('log')
plt.savefig(f'{archive_path}/loss_log.png')
plt.close()

np.savez_compressed(f'{archive_path}/loss.npz', loss=history.history['loss'])

# Evaluating model ------------------------------------------------------------

score = model.evaluate(test_x, test_y, verbose=0)
print("Score:", score)

with open(f'{archive_path}/scores.txt', 'w') as f:
    f.write(f'Score:    \t{score}\n')



#==============================================================================
# PREDICTIONS
#==============================================================================



r = np.random.randint(0, len(x)+1)
x_prediction = np.array([x[r]])
print(x_prediction.shape)

y_prediction = model.predict(x_prediction)[0,0]
print(y_prediction.shape)

print(f"Expected: {y[r] * y_maxs[0]:.2e} Msun")
print(f"Predicted: {y_prediction * y_maxs[0]:.2e} Msun")

np.savez_compressed(f'{archive_path}/prediction.npz', x=x_prediction, y=y_prediction)


