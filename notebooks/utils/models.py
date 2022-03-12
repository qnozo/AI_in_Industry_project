import imp


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as k
import utils.utils as utils
import numpy as np

class MLPRegressor(keras.Model):
    def __init__(self, input_shape, hidden=[]):
        super(MLPRegressor, self).__init__()
        # Build the model
        self.lrs = [layers.Dense(h, activation='relu') for h in hidden]
        self.lrs.append(layers.Dense(1, activation='linear'))

    def call(self, data):
        x = data
        for layer in self.lrs:
            x = layer(x)
        return x



class CstBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, in_cols, batch_size=32, seed=42):
        super(CstBatchGenerator).__init__()
        self.data = data
        self.in_cols = in_cols
        self.dpm = utils.split_by_field(data, 'machine')
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        # Build the first sequence of batches
        self.__build_batches()

    def __len__(self):
        return len(self.batches)

    # def __getitem__(self, index):
    #     idx = self.batches[index]
    #     mcn = self.machines[index]
    #     x = self.data[self.in_cols].loc[idx].values
    #     y = self.data['rul'].loc[idx].values
    #     return x, y


    def __getitem__(self, index):
        idx = self.batches[index]
        # mcn = self.machines[index]
        x = self.data[self.in_cols].loc[idx].values
        y = self.data['rul'].loc[idx].values
        flags = (y != -1)
        info = np.vstack((y, flags, idx)).T
        return x, info

    def on_epoch_end(self):
        self.__build_batches()

    def __build_batches(self):
        self.batches = []
        self.machines = []
        # Randomly sort the machines
        # self.rng.shuffle(mcns)
        # Loop over all machines
        mcns = list(self.dpm.keys())
        for mcn in mcns:
            # Obtain the list of indices
            index = self.dpm[mcn].index
            # Padding
            padsize = self.batch_size - (len(index) % self.batch_size)
            padding = self.rng.choice(index, padsize)
            idx = np.hstack((index, padding))
            # Shuffle
            self.rng.shuffle(idx)
            # Split into batches
            bt = idx.reshape(-1, self.batch_size)
            # Sort each batch individually
            bt = np.sort(bt, axis=1)
            # Store
            self.batches.append(bt)
            self.machines.append(np.repeat([mcn], len(bt)))
        # Concatenate all batches
        self.batches = np.vstack(self.batches)
        self.machines = np.hstack(self.machines)
        # Shuffle the batches
        bidx = np.arange(len(self.batches))
        self.rng.shuffle(bidx)
        self.batches = self.batches[bidx, :]
        self.machines = self.machines[bidx]

class CstRULRegressor(MLPRegressor):
    def __init__(self, input_shape, alpha, beta, maxrul, hidden=[]):
        super(CstRULRegressor, self).__init__(input_shape, hidden)
        # Weights
        self.alpha = alpha
        self.beta = beta
        self.maxrul = maxrul
        # Loss trackers
        self.ls_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.cst_tracker = keras.metrics.Mean(name='cst')

    def train_step(self, data):
        x, info = data
        y_true = info[:, 0:1]
        flags = info[:, 1:2]
        idx = info[:, 2:3]

        with tf.GradientTape() as tape:
            # Obtain the predictions
            y_pred = self(x, training=True)
            # Compute the main loss
            mse = k.mean(flags * k.square(y_pred-y_true))
            # Compute the constraint regularization term
            delta_pred = y_pred[1:] - y_pred[:-1]
            delta_rul = -(idx[1:] - idx[:-1]) / self.maxrul
            deltadiff = delta_pred - delta_rul
            cst = k.mean(k.square(deltadiff))
            loss = self.alpha * mse + self.beta * cst

        # Compute gradients
        tr_vars = self.trainable_variables
        grads = tape.gradient(loss, tr_vars)

        # Update the network weights
        self.optimizer.apply_gradients(zip(grads, tr_vars))

        # Track the loss change
        self.ls_tracker.update_state(loss)
        self.mse_tracker.update_state(mse)
        self.cst_tracker.update_state(cst)
        return {'loss': self.ls_tracker.result(),
                'mse': self.mse_tracker.result(),
                'cst': self.cst_tracker.result()}

    @property
    def metrics(self):
        return [self.ls_tracker,
                self.mse_tracker,
                self.cst_tracker]

class LagDualRULRegressor(MLPRegressor):
    def __init__(self, input_shape, alpha, beta, maxrul, hidden=[]):
        super(LagDualRULRegressor, self).__init__(input_shape, hidden)
        # Weights
        self.alpha = alpha
        self.beta = beta
        self.maxrul = maxrul
        # Loss trackers
        self.ls_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.cst_tracker = keras.metrics.Mean(name='cst')

    def train_step(self, data):
        x, info = data
        y_true = info[:, 0:1]
        flags = info[:, 1:2]
        idx = info[:, 2:3]

        with tf.GradientTape() as tape:
            # Obtain the predictions
            y_pred = self(x, training=True)
            # Compute the main loss
            mse = k.mean(flags * k.square(y_pred-y_true))
            # Compute the constraint regularization term
            delta_pred = y_pred[1:] - y_pred[:-1]
            delta_rul = -(idx[1:] - idx[:-1]) / self.maxrul
            deltadiff = delta_pred - delta_rul
            cst = k.maximum(0.0, deltadiff)
            loss = self.alpha * mse + self.beta * cst

        # Compute gradients
        tr_vars = self.trainable_variables
        grads = tape.gradient(loss, tr_vars)

        # Update the network weights
        self.optimizer.apply_gradients(zip(grads, tr_vars))

        # Track the loss change
        self.ls_tracker.update_state(loss)
        self.mse_tracker.update_state(mse)
        self.cst_tracker.update_state(cst)
        return {'loss': self.ls_tracker.result(),
                'mse': self.mse_tracker.result(),
                'cst': self.cst_tracker.result()}

    @property
    def metrics(self):
        return [self.ls_tracker,
                self.mse_tracker,
                self.cst_tracker]
