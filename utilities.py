import pandas as pd
import matplotlib.pyplot as plt

PRICE_COLUMN_NAME = 'Value'
DATE_COLUMN_NAME = 'DateTime'

TITLE = "Predicted prices"

def convert_to_applicable_types(df):
    df[PRICE_COLUMN_NAME] = pd.to_numeric(df[PRICE_COLUMN_NAME])
    df[DATE_COLUMN_NAME] = pd.to_datetime(df[DATE_COLUMN_NAME])


def configure_plot():
    plt.figure(figsize=(20, 16))
    plt.xlabel("Date", fontsize=20, labelpad=20)
    plt.ylabel("MWh", fontsize=20)
    plt.title(TITLE, fontsize=30, pad=50)
    plt.xticks(fontsize=20, rotation=60)
    plt.yticks(fontsize=20)
