import pandas as pd
import matplotlib.pyplot as plt
import constants
import os


def init_data():
    os.chdir("..")
    train_df = pd.read_csv(rf'{os.getcwd()}\data\trainingData.csv')
    test_df = pd.read_csv(rf'{os.getcwd()}\data\testingData2022.csv')
    _setup_plot_with_data(train_df, test_df)
    return train_df, test_df


def show_plot(df):
    plt.plot(df[constants.DATE_COLUMN_NAME], df[constants.PRICE_COLUMN_NAME])
    plt.show()


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
