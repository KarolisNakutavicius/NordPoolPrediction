import pandas as pd
import matplotlib.pyplot as plt
import constants
import os
import numpy as np
from models.data_type import DataType
from models.data_with_results import DataWithResults
from matplotlib.dates import DateFormatter


def init_samples_and_labels():
    dataDict = init_data()
    samplesAndLabels = dict()
    for key in dataDict:
        samplesAndLabels[key] = convert_to_samples_and_labels(dataDict[key][constants.PRICE_COLUMN_NAME], constants.WINDOW_SIZE)
    return samplesAndLabels


def init_data():
    file_path = os.path.abspath(os.path.dirname(__file__))
    train_df = pd.read_csv(rf'{file_path}\data\2015-2020MERGED.csv')
    validation_df = pd.read_csv(rf'{file_path}\data\2021ValidationDataMERGED.csv')
    test_df = pd.read_csv(rf'{file_path}\data\2021TestDataMERGED.csv')
    _setup_plot_with_data(train_df, validation_df, test_df)

    return {
        DataType.TRAIN: train_df,
        DataType.VALIDATION: validation_df,
        DataType.TEST: test_df
    }


def show_plot(df):
    plt.plot(df[constants.DATE_COLUMN_NAME], df[constants.PRICE_COLUMN_NAME])
    plt.show()


def plot_predictions(used_model, sample, label, start=0, end=100):
    used_model.evaluate(sample, label, verbose=2)
    train_predictions = used_model.predict(sample).flatten()

    x = init_data()[DataType.TEST][constants.DATE_COLUMN_NAME][start:end]
    results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': label})
    plt.plot(x, results['Train Predictions'][start:end], "-b", label="Prognozuojamos reikšmės")
    plt.plot(x, results['Actuals'][start:end], "-r", label="Tikros reikšmės")
    date_form = DateFormatter("%Y-%m-%d %H:%M")
    plt.gca().xaxis.set_major_formatter(date_form)

    plt.legend(loc="upper right", fontsize=20)
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
    return DataWithResults(np.array(samples), np.array(labels))


def _setup_plot_with_data(*args):
    for x in args:
        _convert_to_applicable_types(x)
    _configure_plot()


def _configure_plot():
    plt.figure(figsize=(14, 12))
    plt.xlabel("Data", fontsize=16, labelpad=20)
    plt.ylabel("EUR / MWh", fontsize=16, labelpad=20)
    plt.title(constants.TITLE_PREDICTED_PRICES, fontsize=30, pad=50)
    plt.xticks(fontsize=12, rotation=10)
    plt.yticks(fontsize=12)


def _convert_to_applicable_types(df):
    df[constants.PRICE_COLUMN_NAME] = pd.to_numeric(df[constants.PRICE_COLUMN_NAME])
    df[constants.DATE_COLUMN_NAME] = pd.to_datetime(df[constants.DATE_COLUMN_NAME])
