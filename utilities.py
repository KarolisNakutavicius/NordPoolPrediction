import pandas as pd
import matplotlib.pyplot as plt
import constants
import os
import numpy as np


def init_data():
    os.chdir("..")
    train_df = pd.read_csv(rf'{os.getcwd()}\data\2015-2022DailyMERGED.csv')
    validation_df = pd.read_csv(rf'{os.getcwd()}\data\2021ValidationDataMERGED.csv')
    test_df = pd.read_csv(rf'{os.getcwd()}\data\2021TestDataMERGED.csv')
    _setup_plot_with_data(train_df, validation_df, test_df)
    return train_df, validation_df, test_df


def show_plot(df):
    plt.plot(df[constants.DATE_COLUMN_NAME], df[constants.PRICE_COLUMN_NAME])
    plt.show()


def plot_predictions(used_model, sample, label, start=0, end=100):
    used_model.evaluate(sample, label, verbose=2)
    train_predictions = used_model.predict(sample).flatten()
    results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': label})
    plt.plot(results['Train Predictions'][start:end])
    plt.plot(results['Actuals'][start:end])
    plt.show()


def convert_to_samples_and_labels(df, window_size):
    df_as_np = df.to_numpy()
    samples = []
    labels = []
    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i + window_size]]
        samples.append(row)
        label = df_as_np[i + window_size]
        labels.append(label)
    return np.array(samples), np.array(labels)


def _setup_plot_with_data(*args):
    for x in args:
        _convert_to_applicable_types(x)
    _configure_plot()


def _configure_plot():
    plt.figure(figsize=(20, 16))
    plt.xlabel("Data", fontsize=20, labelpad=20)
    plt.ylabel("MWh", fontsize=20)
    plt.title(constants.TITLE_PREDICTED_PRICES, fontsize=30, pad=50)
    plt.xticks(fontsize=20, rotation=60)
    plt.yticks(fontsize=20)


def _convert_to_applicable_types(df):
    df[constants.PRICE_COLUMN_NAME] = pd.to_numeric(df[constants.PRICE_COLUMN_NAME])
    df[constants.DATE_COLUMN_NAME] = pd.to_datetime(df[constants.DATE_COLUMN_NAME])

