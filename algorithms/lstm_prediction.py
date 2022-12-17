import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.losses import *
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.models import load_model
import constants
from models.data_type import DataType
import utilities


def create_model():
    sequentialModel = Sequential()
    sequentialModel.add(InputLayer((constants.WINDOW_SIZE, 1)))
    sequentialModel.add(LSTM(64))
    sequentialModel.add(Flatten())
    sequentialModel.add(Dense(8, 'relu'))
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


data = utilities.init_samples_and_labels()

# TRAIN NEW MODEL
model = create_model()
model.fit(
    data[DataType.TRAIN].samples,
    data[DataType.TRAIN].labels,
    validation_data=(data[DataType.VALIDATION].samples, data[DataType.VALIDATION].labels),
    epochs=15,
    callbacks=[ModelCheckpoint(constants.LSTM_MODEL_PATH, save_best_only=True)])

# Load Model
# model = load_model(constants.LSTM_MODEL_PATH,
#                    custom_objects={
#                        'MeanAbsoluteError': MeanAbsoluteError(),
#                        'MeanAbsolutePercentageError': MeanAbsolutePercentageError(),
#                        'RootMeanSquaredError': RootMeanSquaredError()})

model.evaluate(data[DataType.TEST].samples, data[DataType.TEST].labels, verbose=2)
utilities.plot_predictions(model, data[DataType.TEST].samples, data[DataType.TEST].labels)