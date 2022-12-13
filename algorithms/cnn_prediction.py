import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from keras.layers import *
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


def train_model():
    sequentialModel = Sequential()
    sequentialModel.add(InputLayer((WINDOW_SIZE, 1)))
    sequentialModel.add(Conv1D(filters=32,
                               kernel_size=5,
                               kernel_regularizer=regularizers.L1L2(l1=1e-3),
                               ))
    sequentialModel.add(Flatten())
    sequentialModel.add(Dense(1,
                              'relu',
                              ))
    print(sequentialModel.summary())
    cp = ModelCheckpoint(MODEL_PATH, save_best_only=True)
    sequentialModel.compile(
        loss=MeanAbsoluteError(),
        optimizer=Adam(learning_rate=0.00001),
        metrics=[
            MeanAbsoluteError(),
            MeanAbsolutePercentageError(),
            RootMeanSquaredError()
        ])

    sequentialModel.fit(samples_train, label_train, validation_data=(samples_validate, label_validate), epochs=50,
                        callbacks=[cp])
    return sequentialModel


train_df, validation_df, test_df = utilities.init_data()
samples_train, label_train = utilities.convert_to_samples_and_labels(train_df[constants.PRICE_COLUMN_NAME], WINDOW_SIZE)
samples_validate, label_validate = utilities.convert_to_samples_and_labels(validation_df[constants.PRICE_COLUMN_NAME],
                                                                           WINDOW_SIZE)
samples_test, label_test = utilities.convert_to_samples_and_labels(test_df[constants.PRICE_COLUMN_NAME], WINDOW_SIZE)

# TRAIN NEW MODEL
# model = train_model()

# Load Model
model = load_model(MODEL_PATH,
                   custom_objects={
                       'MeanAbsoluteError': MeanAbsoluteError(),
                       'MeanAbsolutePercentageError': MeanAbsolutePercentageError(),
                       'RootMeanSquaredError': RootMeanSquaredError()})

model.evaluate(samples_test, label_test, verbose=2)

# utilities.plot_predictions(model, samples_test, label_test)
