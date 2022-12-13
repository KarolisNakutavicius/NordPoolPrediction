import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# from tensorflow import keras
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.losses import *
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.models import load_model
from DataType import DataType
import utilities

WINDOW_SIZE = 5
MODEL_PATH = 'lstm_model/'


def create_model():
    sequentialModel = Sequential()
    sequentialModel.add(InputLayer((WINDOW_SIZE, 1)))
    sequentialModel.add(LSTM(32))
    sequentialModel.add(Flatten())
    sequentialModel.add(Dense(1, 'relu'))
    sequentialModel.compile(
        loss=MeanAbsoluteError(),
        optimizer=Adam(learning_rate=0.0001),
        metrics=[
            MeanAbsoluteError(),
            MeanAbsolutePercentageError(),
            RootMeanSquaredError()
        ])
    return sequentialModel


data = utilities.init_samples_and_labels(WINDOW_SIZE)

# TRAIN NEW MODEL
model = create_model()
model.fit(
    data[DataType.TEST].samples,
    data[DataType.TEST].labels,
    validation_data=(data[DataType.VALIDATION].samples, data[DataType.VALIDATION].labels),
    epochs=50,
    callbacks=[ModelCheckpoint(MODEL_PATH, save_best_only=True)])

# Load Model
# model = load_model(MODEL_PATH,
#                    custom_objects={
#                        'MeanAbsoluteError': MeanAbsoluteError(),
#                        'MeanAbsolutePercentageError': MeanAbsolutePercentageError(),
#                        'RootMeanSquaredError': RootMeanSquaredError()})

model.evaluate(data[DataType.TEST].samples, data[DataType.TEST].labels, verbose=2)
utilities.plot_predictions(model, data[DataType.TEST].samples, data[DataType.TEST].labels)