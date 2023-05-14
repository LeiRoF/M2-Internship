import os
import numpy as np
import tensorflow as tf
import keras.backend as K
from LRFutils import progress, logs
from time import time
import matplotlib.pyplot as plt
from . import sysinfo
import json
import yaml
from .dataset import *

class Model(tf.keras.models.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    #==========================================================================
    # TRAINING & PREDICTION
    #==========================================================================

    # Train model -------------------------------------------------------------

    def fit(self, train, epochs, batch_size, val=None, verbose=True, plot_loss=False):
        
        logs.info("Training model...")

        # Progress bar callback ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        class CustomCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                self.bar = progress.Bar(epochs)
                self.last_update = time()
                self.loss_value = []
                self.loss_epoch = []
                if plot_loss:
                    plt.ion()
                    self.fig = plt.figure()
                    self.ax = self.fig.add_subplot(111)
                    self.loss_curve, = self.ax.plot(self.loss_value, self.loss_epoch, 'r-')

            def on_epoch_end(self, epoch, logs=None):
                self.bar(epoch+1, prefix = f"Loss: {logs['loss']:.2e} | {sysinfo.get()}")
                if plot_loss and ((time() - self.last_update) > 1):
                    self.loss_value.append(logs['loss'])
                    self.loss_epoch.append(epoch)
                    self.loss_curve.set_xdata(self.loss_epoch)
                    self.loss_curve.set_ydata(self.loss_value)
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()

        # Training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
        start_time = time()

        train_x = dict(train.filter(self.input_names))
        train_y = dict(train.filter(self.output_names))
        val_x = dict(val.filter(self.input_names))
        val_y = dict(val.filter(self.output_names))

        history = super().fit(train_x, train_y,
                            epochs=epochs,
                            batch_size=batch_size, 
                            validation_data=(val_x, val_y), 
                            verbose=0,
                            callbacks=[CustomCallback()],
                            workers=1,
                            use_multiprocessing=True
                        )

        training_time = time() - start_time

        logs.info("Model trained. ✅")

        return history, training_time
    
    # Predictions -------------------------------------------------------------
    
    def predict(self, test_dataset, *args, display=False, save_as=None, N=1, **kwargs):

        test_x = dict(test_dataset.filter(self.input_names))
        test_y = dict(test_dataset.filter(self.output_names))

        p = []
        for _ in range(N):
            p.append(super().predict(test_x, *args, verbose=0, **kwargs))

        results = {}
        for label in self.output_names:
            results[label] = []
            for e, expectation in enumerate(test_y[label]):
                results[label].append({"expectation": expectation,"predictions":[]})
                for prediction in p:
                    results[label][e]["predictions"].append(prediction[label][e])

        i = 0
        if display:
            for output in results.keys():
                # print(output)
                for run in results[output]:
                    # print(run)
                    expectation = run["expectation"]
                    print(
                        f"   Epxpectation {i} : ",
                        expectation.shape,
                        f"Mean: {np.mean(expectation):.2e}",
                        f"Std: {np.std(expectation):.2e}",
                        f"Min: {np.min(expectation):.2e}",
                        f"Max: {np.max(expectation):.2e}"
                    )
                    i+=1

                    for j, prediction in enumerate(run["predictions"]):
                        print(
                            f"      Prediction {j} : ",
                            prediction.shape,
                            f"Mean: {np.mean(prediction):.2e}",
                            f"Std:{np.std(prediction):.2e}",
                            f"Min:{np.min(prediction):.2e}",
                            f"Max:{np.max(prediction):.2e}"
                        )

        # Save predictions
        if save_as is not None:
            np.savez_compressed(save_as,
                inference=results,
            )

        return results

    #==========================================================================
    # MODEL REPRESENTATIONS
    #==========================================================================

    # Model summary -----------------------------------------------------------

    def summary(self):
        stringlist = []
        super().summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        return short_model_summary

    def print(self):
        print(self.summary())

    # Plot a model representation ---------------------------------------------

    def plot(self, archive_path):
        trainable_count = np.sum([K.count_params(w) for w in self.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in self.non_trainable_weights])

        return tf.keras.utils.plot_model(
            self,
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
    
    #==========================================================================
    # SAVE MODEL & METADATA
    #==========================================================================
    
    # Archive model & metadata ------------------------------------------------

    def save(self, archive_path, history=None, **metadata):

        super().save(os.path.join(archive_path,"model.h5"))

        self.plot(archive_path)

        summary = self.summary().split("\n")

        # History treatment ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if history is not None:
            if len(self.output_names) == 1:
                output_name = self.output_names[0]
                keys = list(history.history.keys())
                for key in keys:
                    if key != "loss" and key != "val_loss":
                        if key.startswith("val_"):
                            new_key = "val_" + output_name + "_" + key[4:]
                        else:
                            new_key = output_name + "_" + key
                        history.history[new_key] = history.history[key]
                        del history.history[key]
            
            N = len(history.history)
            N1 = int(np.sqrt(N))
            N2 = N1
            if N1*N2 < N:
                N2 += 1
            if N1*N2 < N:
                N1 += 1

            fig, axs = plt.subplots(N1, N2, figsize=(N2*5, N1*5))
            axs = axs.flatten()
            i = 0
            for key, value in history.history.items():
                if key.startswith("val_"):
                    continue
                axs[i].plot(value, label=key)
                axs[i].plot(history.history[f"val_{key}"], label=f"val_{key}")
                axs[i].set_title(key)
                axs[i].legend()
                axs[i].set_xlabel("Epoch")
                axs[i].set_ylabel(key)
                i += 1

                axs[i].plot(value, label=key)
                axs[i].plot(history.history[f"val_{key}"], label=f"val_{key}")
                axs[i].set_title(key + " in log:log scale")
                axs[i].legend()
                axs[i].set_xlabel("Epoch")
                axs[i].set_ylabel(key)
                axs[i].set_yscale("log")
                axs[i].set_xscale("log")    
                i += 1

            fig.savefig(f"{archive_path}/history.png")

        # Save metadata ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        np.savez_compressed(f"{archive_path}/metadata.npz", history=history.history, summary=summary, **metadata)

    # Save reference ----------------------------------------------------------

    def save_reference(self, reference_path, archive_path):

        # Create reference folder ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if not os.path.isdir(os.path.join(reference_path, "output")):
            os.makedirs(os.path.join(reference_path, "output"))
        if not os.path.isdir(os.path.join(reference_path, "problem")):
            os.makedirs(os.path.join(reference_path, "problem"))

        # Getting mode id ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        try:
            with open(os.path.join(reference_path, "model_list.yml"), "r") as f:
                models = yaml.safe_load(f)

            last_model = int(list(models.keys())[-1], 16) # get last model id from hexa
            new_model = last_model + 1
            new_model = hex(new_model)[2:] # convert to hexa
            new_model = "0"*(4-len(new_model)) + new_model # pad with 0 to reach 4 hexa digits

        except FileNotFoundError:
            models = {}
            new_model = "0000"

        # Save model reference ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        models[str(new_model)] = archive_path

        # Global list
        with open(os.path.join(reference_path, "model_list.yml"), "w") as f:
            yaml.dump(models, f)

        # Output list
        for output in self.output_names:
            with open(os.path.join(reference_path, f"output/{output}.yml"), "a") as f:
                f.write(f"'{new_model}': {archive_path}\n")

        # Problem list
        problem = ",".join(self.input_names) + "---" + ",".join(self.output_names)
        with open(os.path.join(reference_path, f"problem/{problem}.yml"), "a") as f:
            f.write(f"'{new_model}': {archive_path}\n")

        return new_model
