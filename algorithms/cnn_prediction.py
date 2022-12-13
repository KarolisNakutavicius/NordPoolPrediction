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
from keras import regularizers
import utilities
from models.data_type import DataType


def create_model():
    sequentialModel = Sequential()
    sequentialModel.add(InputLayer((constants.WINDOW_SIZE, 1)))
    sequentialModel.add(Conv1D(filters=32,
                               kernel_size=5,
                               kernel_regularizer=regularizers.L1L2(l1=1e-3),
                               ))
    sequentialModel.add(Flatten())
    sequentialModel.add(Dense(1,
                              'relu',
                              ))
    print(sequentialModel.summary())
    sequentialModel.compile(
        loss=MeanAbsoluteError(),
        optimizer=Adam(learning_rate=0.00001),
        metrics=[
            MeanAbsoluteError(),
            MeanAbsolutePercentageError(),
            RootMeanSquaredError()
        ])

    return sequentialModel


data = utilities.init_samples_and_labels()

# TRAIN NEW MODEL
# model = create_model()
# model.fit(
#     data[DataType.TEST].samples,
#     data[DataType.TEST].labels,
#     validation_data=(data[DataType.VALIDATION].samples, data[DataType.VALIDATION].labels),
#     epochs=50,
#     callbacks=[ModelCheckpoint(constants.CNN_MODEL_PATH, save_best_only=True)])

# Load Model
model = load_model(constants.CNN_MODEL_PATH,
                   custom_objects={
                       'MeanAbsoluteError': MeanAbsoluteError(),
                       'MeanAbsolutePercentageError': MeanAbsolutePercentageError(),
                       'RootMeanSquaredError': RootMeanSquaredError()})

model.evaluate(data[DataType.TEST].samples, data[DataType.TEST].labels, verbose=2)
utilities.plot_predictions(model, data[DataType.TEST].samples, data[DataType.TEST].labels)
