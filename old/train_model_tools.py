# Imports ---------------------------------------------------------------------

import os
# # TF config
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from LRFutils import archive, progress, color
from multiprocess import Pool, cpu_count
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy as copy
import pickle
from time import time
import json
import GPUtil
from keras.layers import Input, Dense, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Reshape, Conv3DTranspose, Flatten, Concatenate, Dropout
from keras.models import Model
import keras.backend as K
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Global variables ------------------------------------------------------------

archive_path = archive.new(verbose=True)

try:
    ncpu = cpu_count()
except:
    with open(os.getenv("OAR_NODEFILE"), 'r') as f:
        ncpu = len(f.readlines())

# Functions -------------------------------------------------------------------

def system_info():

    return f"CPU: {psutil.cpu_percent()}%"\
        + f", GPU: {np.mean([i.load for i in GPUtil.getGPUs()])*100:.1f}%"\
        + f", RAM: {psutil.virtual_memory().percent}%"\
        + f" ({psutil.virtual_memory().used/1024**3:.1f}GB"\
        + f"/{psutil.virtual_memory().total/1024**3:.1f}GB)"

def nb_vec(x:dict) -> int:

    for key in x.keys():
        assert len(x[key]) == len(x[list(x.keys())[0]]), "All dictionary element must be the same length"

    return len(x[list(x.keys())[0]])

def get_sample(x):
    sample = {}
    for key, value in x.items():
        sample[key] = value[0]
    return sample

def pick_vec(x, y, idx):
    vec_x = {}
    for key, value in x.items():
        vec_x[key] = value[idx]
    
    vec_y = {}
    for key, value in y.items():
        vec_y[key] = value[idx]

    return vec_x, vec_y

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
# Load data -------------------------------------------------------------------

# Loop over files

def load_data(load_file, dataset_path) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Do what you want int this function, as long as it returns the following:
    - list[3D-ndarray] : input vectors
    - list[3D-ndarray] : output vectors
    """

    if not is_notebook():
        print("Loading data...")
    
    # Limit of the number of vectors to load
    max_files = 1000
    files = os.listdir(dataset_path)
    nb_vectors = min(len(files), max_files)

    # Load data
    x = {}
    y = {}
    bar = progress.Bar(nb_vectors, "Loading data")
    for i, file in enumerate(files):
        if i >= nb_vectors:
            break
        bar(i, prefix=system_info())
        
        new_x, new_y = load_file(f"{dataset_path}/{file}")

        for key, value in new_x.items():
            if key not in x:
                x[key] = []
            x[key].append(value)

        for key, value in new_y.items():
            if key not in y:
                y[key] = []
            y[key].append(value)

    
    for key in x.keys():
        x[key] = np.array(x[key])
    
    for key in y.keys():
        y[key] = np.array(y[key])
        
    bar(nb_vectors) 
    print(f"Number of vectors: {nb_vectors}")

    return x, y, nb_vectors

def filter_data(model, x, y):
    x = copy(x)
    y = copy(y)
    for key in list(x.keys()):
        if key.replace(" ", "_").lower() not in [i.name.replace(" ", "_").lower().split("/")[0] for i in model.inputs]:
            del x[key]
    for key in list(y.keys()):
        if key.replace(" ", "_").lower() not in [o.name.replace(" ", "_").lower().split("/")[0] for o in model.outputs]:
            del y[key]
    return x, y

def get_labels(x,y):
    x_labels = list(x.keys())
    y_labels = list(y.keys())

    return x_labels, y_labels

def get_shapes(x, y):
    x_shapes = [i[0].shape for i in x.values()]
    y_shapes = [i[0].shape for i in y.values()]
    return x_shapes, y_shapes

def print_shapes(x_labels, y_labels, x_shapes, y_shapes):
    print(f"X shapes:\n -", '\n - '.join([f"{i}: {j}" for i, j in zip(x_labels, x_shapes)]))
    print(f"Y shapes:\n -", '\n - '.join([f"{i}: {j}" for i, j in zip(y_labels, y_shapes)]))  

# Show 10 random vectors ------------------------------------------------------

def show_random_vectors(x, y):
    # Show 10 random vectors

    r = np.array(np.random.randint(0, nb_vec(x), 10))
    x_print, y_print = pick_vec(x, y, r)

    if is_notebook():

        nb_axs = 0
        if "Dust map" in x: nb_axs += 1
        if "CO cube" in x: nb_axs += 1
        if "N2H cube" in x: nb_axs += 1

        fig, axs = plt.subplots(nb_axs, 10, figsize=(15, 5))
        if nb_axs == 1: axs = np.array([axs])

        for i in range(10):
            vec = r[i]
            nb_axis = 0
            if "Dust map" in x:
                axs[nb_axis, i].imshow(x["Dust map"][vec])
                axs[nb_axis, i].set_title(f"Dust {vec}")
                nb_axis += 1
            if "CO cube" in x:
                axs[nb_axis, i].imshow(np.sum(x["CO cube"][vec], axis=(-1,-2)))
                axs[nb_axis, i].set_title(f"CO {vec}")
                nb_axis += 1
            if "N2H cube" in x:
                axs[nb_axis, i].imshow(np.sum(x["N2H cube"][vec], axis=(-1,-2)))
                axs[nb_axis, i].set_title(f"N2H+ {vec}")
                nb_axis += 1

    return pd.DataFrame(
        {key + " max": [np.max(value[i]) for i in r]  for key, value in (x_print | y_print).items()} | {key + " min": [np.min(value[i]) for i in r] for key, value in (x_print | y_print).items()}
        , index=r)
    
# Post processing -------------------------------------------------------------

# Normalization

def normalize(x):

    normalized = copy(x)
    means = {}
    stds = {}
    for key, value in x.items():
        means[key] = np.mean(value)
        stds[key] = np.std(value)
        normalized[key] = (value - means[key]) / stds[key]

    return normalized, means, stds

# Print means and stds using pandas

def print_means_stds(means, stds):
    return pd.DataFrame([means, stds], index=["Mean", "Std"])

# Shuffle and split data ------------------------------------------------------

def shuffle(*args):

    idx = np.random.permutation(nb_vec(args[0]))

    args = copy(args)

    for x in args:

        x_copy = copy(x)
        for key in x.keys():
            x_copy[key] = x[key][idx]

        x = x_copy

    return args

# Splitting dataset

def split(x, y, valid_frac=0.2, test_frac=0.1):

    nb_vectors = nb_vec(x)
    train_frac = 1 - valid_frac - test_frac
    
    train_x, valid_x, test_x = {}, {}, {}

    for key, value in x.items():
        train_x[key] = value[:int(nb_vectors*train_frac)]
        valid_x[key] = value[int(nb_vectors*train_frac):int(nb_vectors*(train_frac+valid_frac))]
        test_x[key] = value[int(nb_vectors*(train_frac+valid_frac)):]

    train_y, valid_y, test_y = {}, {}, {}

    for key, value in y.items():
        train_y[key] = value[:int(nb_vectors*train_frac)]
        valid_y[key] = value[int(nb_vectors*train_frac):int(nb_vectors*(train_frac+valid_frac))]
        test_y[key] = value[int(nb_vectors*(train_frac+valid_frac)):]

    return train_x, train_y, valid_x, valid_y, test_x, test_y

# Show -------------------------------------------------------------------------

def get_summary(model):
    # Store and print model summary
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    return short_model_summary


def print_model(model):

    if is_notebook():

        trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

        print('Total params: {:,}'.format(trainable_count + non_trainable_count))
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))

        return tf.keras.utils.plot_model(
            model,
            to_file=f"{archive_path}/model.png",
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96,
            layer_range=None,
            show_layer_activations=True,
        )

    else:
        return get_summary(model)

# Train model -----------------------------------------------------------------

def train(model, train_x, train_y, valid_x, valid_y, test_x, test_y, epochs, batch_size):

    class CustomCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.bar = progress.Bar(epochs)
            self.last_update = time()
            self.loss_value = []
            self.loss_epoch = []
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.loss_curve, = self.ax.plot(self.loss_value, self.loss_epoch, 'r-')

        def on_epoch_end(self, epoch, logs=None):
            self.bar(epoch+1, prefix = f"Loss: {logs['loss']:.2e} | {system_info()}")
            if (time() - self.last_update) > 1:
                self.loss_value.append(logs['loss'])
                self.loss_epoch.append(epoch)
                self.loss_curve.set_xdata(self.loss_epoch)
                self.loss_curve.set_ydata(self.loss_value)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
   
    start_time = time()

    history = model.fit(train_x, train_y,
                        epochs=epochs,
                        batch_size=batch_size, 
                        validation_data=(valid_x, valid_y), 
                        verbose=0,
                        callbacks=[CustomCallback()],
                        workers=10,
                        use_multiprocessing=True)

    training_time = time() - start_time

    model.save(f'{archive_path}/model0.h5')
    with open(f'{archive_path}/history.pickle', "wb") as file_pi:
        pickle.dump(history.history, file_pi)

    score = model.evaluate(test_x, test_y, verbose=0)
    print("Score:", score)

    return history, training_time, score

# Plot history ----------------------------------------------------------------

def plot_history(history):
    # Show trining history

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    for key in history.history.keys():
        if (not key.startswith('val_')) and (key.endswith('loss')):
            plt.plot(history.history[key], alpha=0.5, label=key.replace(" ", ""))
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(2, 2, 2)
    for key in history.history.keys():
        if (not key.startswith('val_')) and (key.endswith('loss')):
            plt.plot(history.history[key], alpha=0.5, label=key.replace(" ", ""))
    plt.title('Model loss in log:log scale')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(2, 2, 3)
    for key in history.history.keys():
        if (key.startswith('val_')) and (not key.endswith('_loss')):
            plt.plot(history.history[key], alpha=0.5, label=key.replace(" ", ""))
    plt.title('Model metrics')
    plt.ylabel('Metric')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(2, 2, 4)
    for key in history.history.keys():
        if (key.startswith('val_')) and (not key.endswith('_loss')):
            plt.plot(history.history[key], alpha=0.5, label=key.replace(" ", ""))
    plt.title('Model metrics in log:log scale')
    plt.ylabel('Metric')
    plt.xlabel('Epoch')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')

    plt.show()

# Save details ----------------------------------------------------------------

def save_details(summary, loss, optimizer, metrics, valid_frac, test_frac, epochs, batch_size, training_time, score, nb_vectors, x_labels, y_labels):
    # Save details to json file
    with open(f"{archive_path}/model_details.json", "w") as fp:

        json.dump({
            "summary": summary.split("\n"),
            "loss": loss,   
            "optimizer": optimizer,
            "metrics": [str(i) for i in metrics],
            "valid_frac": valid_frac,
            "test_frac": test_frac,
            "epochs": epochs,
            "batch_size": batch_size,
            "training_time": training_time,
            "score": score,
            "dataset_size": nb_vectors,
            "path": archive_path,
            "inputs": x_labels,
            "outputs": y_labels,
        }, fp, indent=4)

# Add record ------------------------------------------------------------------

def add_record(x_labels, y_labels):
    # Add record to comparison file

    if input("Keep this training in records? (Y/n): ").lower() in ["","y","yes"]:

        for label in y_labels:
            file_path = f"data/model_comparison/{label.replace(' ','_')}.yml"

            with open(file_path, "a") as text_file:
                    text_file.write(f"- {archive_path}\n")

        # Detailed

        inputs_txt = ','.join(x_labels).replace(' ', '_')
        outputs_txt = ','.join(y_labels).replace(' ', '_')

        file_path = f"data/model_comparison/detailed/{inputs_txt}---{outputs_txt}.yml"

        with open(file_path, "a") as text_file:
                text_file.write(f"- {archive_path}\n")