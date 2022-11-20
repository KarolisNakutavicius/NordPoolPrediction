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
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.models import load_model
import utilities
import constants


def convert_to_samples_and_labels(df, window_size=5):
    df_as_np = df.to_numpy()
    samples = []
    labels = []
    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i + window_size]]
        samples.append(row)
        label = df_as_np[i + window_size]
        labels.append(label)
    return np.array(samples), np.array(labels)


train_df, test_df = utilities.init_data()

train_df.index = train_df[constants.DATE_COLUMN_NAME]
temp = train_df[constants.PRICE_COLUMN_NAME]

samples, label = convert_to_samples_and_labels(temp)

X_train, y_train = samples[:6000], label[:6000]
# X_val, y_val = samples[6000:6500], label[6000:6500]
X_test, y_test = samples[6500:], label[6500:]

# model1 = Sequential()
# model1.add(InputLayer((5, 1))) # Or keras.layers.Flatten()
# model1.add(LSTM(64))  # 64?? Define meaning
# model1.add(Dense(8, 'relu'))
# model1.add(Dense(1, 'linear'))  #Define
#
# print(model1.summary())

# cp = ModelCheckpoint('model2/', save_best_only=True)
# model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
#
# model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, callbacks=[cp]) # Diffs between validtion and train data


model1 = load_model('model1/')

train_predictions = model1.predict(X_test).flatten()
model1.evaluate(X_test, y_test, verbose=2)

train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_test})

print(train_results)
#
plt.plot(train_results['Train Predictions'][:5])
plt.plot(train_results['Actuals'][:5])
plt.show()


# train_predictions = model1.predict(X_val).flatten()
# train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_val})
#
# print(train_results)
#
# plt.plot(train_results['Train Predictions'][:100])
# plt.plot(train_results['Actuals'][:100])
# plt.show()

# train_predictions = model1.predict(X_test).flatten()
# train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_test})
#
# print(train_results)
#
# plt.plot(train_results['Train Predictions'][:100])
# plt.plot(train_results['Actuals'][:100])
# plt.show()
