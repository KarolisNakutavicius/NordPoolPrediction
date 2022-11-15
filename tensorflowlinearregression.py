import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import matplotlib.dates as dates
# from matplotlib.pyplot import figure
from matplotlib import pyplot as plt


def convert_to_applicable_types(df):
    df[PRICE_COLUMN_NAME] = pd.to_numeric(df[PRICE_COLUMN_NAME])
    df[DATE_COLUMN_NAME] = pd.to_datetime(df[DATE_COLUMN_NAME])


PRICE_COLUMN_NAME = 'Value'
DATE_COLUMN_NAME = 'DateTime'

train_df = pd.read_csv(rf'{os.getcwd()}\trainingData.csv')
test_df = pd.read_csv(rf'{os.getcwd()}\testingData2022.csv')

# print(type(dates.datestr2num(test_df[0][DATE_COLUMN_NAME])))

convert_to_applicable_types(train_df)
convert_to_applicable_types(test_df)

plt.figure(figsize=(20, 16))

plt.plot(test_df[DATE_COLUMN_NAME], test_df[PRICE_COLUMN_NAME])

plt.xlabel("Date", fontsize=20, labelpad=20)
plt.ylabel("MWh", fontsize=20)
plt.title("Actual Values", fontsize=30, pad=50)
plt.xticks(fontsize=20, rotation=60)
plt.yticks(fontsize=20)
plt.show()

print(test_df.tail())
