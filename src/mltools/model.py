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

class Model(tf.keras.models.Model):

    def __init__(self, *args, dataset, verbose, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset.filter(self.input_names, self.output_names)

        if verbose:
            self.print()
            print(self.dataset)

    def summary(self):
        # Store and print model summary
        stringlist = []
        super().summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        return short_model_summary

    def print(self):
        print(self.summary())

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

    def save(self, archive_path, history=None, **kwargs):
        try:
            super().save(os.path.join(archive_path,"model.h5"))
        except OSError as e:
            logs.warn(f"Could not save model due to the following OSError:{e}")
        self.plot(archive_path)

        summary = self.summary().split("\n")

        dic = dict(summary=summary, **kwargs)

        json.dump(dic, open(f'{archive_path}/details.json', 'w'), indent=4)

        if history is not None:
            np.savez_compressed(f'{archive_path}/history.npz', **history.history, allow_pickle=True)
            
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

    def save_reference(self, reference_path, archive_path):

        if not os.path.isdir(os.path.join(reference_path, "output")):
            os.makedirs(os.path.join(reference_path, "output"))
        if not os.path.isdir(os.path.join(reference_path, "problem")):
            os.makedirs(os.path.join(reference_path, "problem"))

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

        models[new_model] = archive_path

        with open(os.path.join(reference_path, "model_list.yml"), "w") as f:
            yaml.dump(models, f)

        for output in self.output_names:
            with open(os.path.join(reference_path, f"output/{output}.yml"), "a") as f:
                f.write(f"'{new_model}': {archive_path}\n")

        problem = ",".join(self.input_names) + "---" + ",".join(self.output_names)
        with open(os.path.join(reference_path, f"problem/{problem}.yml"), "a") as f:
            f.write(f"'{new_model}': {archive_path}\n")

        return new_model

    def fit(self, epochs, batch_size, verbose=True, plot_loss=False):
        
        logs.info("Training model...")

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
                self.bar(epoch+1, prefix = f"Loss: {logs['loss']:.5f} | {sysinfo.get()}")
                if plot_loss and ((time() - self.last_update) > 1):
                    self.loss_value.append(logs['loss'])
                    self.loss_epoch.append(epoch)
                    self.loss_curve.set_xdata(self.loss_epoch)
                    self.loss_curve.set_ydata(self.loss_value)
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
    
        start_time = time()

        history = super().fit(self.dataset.train.x, self.dataset.train.y,
                            epochs=epochs,
                            batch_size=batch_size, 
                            validation_data=(self.dataset.val.x, self.dataset.val.y), 
                            verbose=0,
                            callbacks=[CustomCallback()],
                            # workers=10,
                            # use_multiprocessing=True
                        )

        training_time = time() - start_time

        scores = self.evaluate(self.dataset.test.x, self.dataset.test.y, verbose=0)

        logs.info("Model trained. ✅")

        return history, training_time, scores
    
    def predict(self, x, *args, display=False, save_as=None, **kwargs):
        y_prediction = super().predict(x, **kwargs)

        y_predicted = {}
        y_expected = {}

        for key, value in y_prediction.items():
            y_predicted[key] = []
            y_expected[key] = []

            if display:
                print(f"   {key} :")
            for i, prediction in enumerate(value):
                p = prediction.flatten()[0] * self.dataset.ystds[key] + self.dataset.ymeans[key]
                y_predicted[key].append(p)

                y = self.dataset.test.y[key][i].flatten()[0] * self.dataset.ystds[key] + self.dataset.ymeans[key]
                y_expected[key].append(y)

                if display:
                    r = self.dataset.ystds[key]
                    print(f"      {i} : Predicted: {p:.2e}, Expected: {y:.2e}, Error: {(p-y):.2e} ({np.abs(p-y)/r:.2f} σ)")

        # Save predictions
        if save_as is not None:
            np.savez_compressed(save_as,
                prediction=y_predicted,
                expected=y_expected,
                mean=self.dataset.ymeans,
                sigma=self.dataset.ystds,
            )

        return y_predicted, y_expected

