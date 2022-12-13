import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# from tensorflow import keras
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.losses import *
from keras.metrics import RootMeanSquaredError
from keras.models import load_model
from keras.optimizers import Adam
from keras import regularizers
import constants
import utilities
from models.data_type import DataType


def create_model():
    sequentialModel = Sequential()
    sequentialModel.add(InputLayer((constants.WINDOW_SIZE, 1)))
    sequentialModel.add(GRU(32,
                  kernel_regularizer=regularizers.L1(l1=0.001),
                  # bias_regularizer=regularizers.L2(1e-4),
                  # activity_regularizer=regularizers.L2(1e-5)
                  ))
    sequentialModel.add(Flatten())
    sequentialModel.add(Dense(1, 'relu',
                    # kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                    # bias_regularizer=regularizers.L2(1e-4),
                    # activity_regularizer=regularizers.L2(1e-5)
                    ))
    # model.add(Dense(1, 'linear'))
    sequentialModel.compile(
        loss=MeanAbsoluteError(),
        optimizer=Adam(learning_rate=0.001),
        metrics=[
            MeanAbsoluteError(),
            MeanAbsolutePercentageError(),
            RootMeanSquaredError()
        ])
    print(sequentialModel.summary())
    return sequentialModel


data = utilities.init_samples_and_labels()

# TRAIN NEW MODEL
model = create_model()
model.fit(
    data[DataType.TEST].samples,
    data[DataType.TEST].labels,
    validation_data=(data[DataType.VALIDATION].samples, data[DataType.VALIDATION].labels),
    epochs=50,
    callbacks=[ModelCheckpoint(constants.GRU_MODEL_PATH, save_best_only=True)])

# Load Model
# model = load_model(constants.GRU_MODEL_PATH,
#                    custom_objects={
#                        'MeanAbsoluteError': MeanAbsoluteError(),
#                        'MeanAbsolutePercentageError': MeanAbsolutePercentageError(),
#                        'RootMeanSquaredError': RootMeanSquaredError()})

utilities.plot_predictions(model, data[DataType.TEST].samples, data[DataType.TEST].labels)