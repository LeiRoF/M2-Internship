import os
import numpy as np
import tensorflow as tf
import keras.backend as K
from LRFutils import progress
from time import time
import matplotlib.pyplot as plt
from . import sysinfo

class Model(tf.keras.models.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        print('Total params: {:,}'.format(trainable_count + non_trainable_count))
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))

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

    def save(self, archive_path, history=None):
        super().save(os.path.join(archive_path,"model.h5"))
        with open(os.path.join(archive_path, "model_summary.txt"), "w") as text_file:
            text_file.write(self.summary())
            self.plot(archive_path)


        if history is not None:
            np.savez_compressed(f'{archive_path}/history.npz', **history.history, allow_pickle=True)

    def fit(self, dataset, epochs, batch_size, verbose=True, plot_loss=False):

        dataset = dataset.filter(self.input_names, self.output_names)

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
    
        start_time = time()

        print(dataset.train.x.keys())
        print(dataset.train.y.keys())
        print(dataset.val.x.keys())
        print(dataset.val.y.keys())
        print(len(dataset.train))
        print(len(dataset.val))

        history = super().fit(dataset.train.x, dataset.train.y,
                            epochs=epochs,
                            batch_size=batch_size, 
                            validation_data=(dataset.val.x, dataset.val.x), 
                            verbose=0,
                            callbacks=[CustomCallback()],
                            workers=10,
                            use_multiprocessing=True
                        )

        training_time = time() - start_time

        return history, training_time
