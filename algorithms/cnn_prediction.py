import os

import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras
import numpy as np
import tensorflow as tf
# from tensorflow import keras
from keras.layers import *
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.losses import *
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.models import load_model
from keras import regularizers
import utilities
import constants

WINDOW_SIZE = 5
MODEL_PATH = 'cnn_model/'


def convert_to_samples_and_labels(df, window_size=WINDOW_SIZE):
    df_as_np = df.to_numpy()
    samples = []
    labels = []
    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i + window_size]]
        samples.append(row)
        label = df_as_np[i + window_size]
        labels.append(label)
    return np.array(samples), np.array(labels)


def plot_predictions(used_model, sample, label, start=0, end=100):
    train_predictions = used_model.predict(sample).flatten()
    used_model.evaluate(sample, label, verbose=2)
    results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': label})
    plt.plot(results['Train Predictions'][start:end])
    plt.plot(results['Actuals'][start:end])
    plt.show()


def train_model():
    model = Sequential()
    model.add(InputLayer((WINDOW_SIZE, 1)))
    model.add(Conv1D(filters=32,
                     kernel_size=5,
                     kernel_regularizer=regularizers.L1L2(l1=1e-3),
                     ))
    model.add(Flatten())
    model.add(Dense(1,
                    'relu',
                    ))
    print(model.summary())
    cp = ModelCheckpoint(MODEL_PATH, save_best_only=True)
    model.compile(
        loss=MeanAbsoluteError(),
        optimizer=Adam(learning_rate=0.00001),
        metrics=[
            MeanAbsoluteError(),
            MeanAbsolutePercentageError(),
            RootMeanSquaredError()
        ])

    model.fit(samples_train, label_train, validation_data=(samples_test, label_test), epochs=50, callbacks=[cp])
    return model


train_df, test_df = utilities.init_data()

samples_train, label_train = convert_to_samples_and_labels(train_df[constants.PRICE_COLUMN_NAME])
samples_test, label_test = convert_to_samples_and_labels(test_df[constants.PRICE_COLUMN_NAME])

# TRAIN NEW MODEL
model = train_model()

# Load Model
# model = load_model(MODEL_PATH,
#                    custom_objects={
#                        'MeanAbsoluteError': MeanAbsoluteError(),
#                        'MeanAbsolutePercentageError': MeanAbsolutePercentageError(),
#                        'RootMeanSquaredError': RootMeanSquaredError()})4


model.evaluate(samples_test, label_test, verbose=2)

# utilities.show_plot(train_df)

# plot_predictions(model, samples_test, label_test)
