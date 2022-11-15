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

train_df = pd.read_csv(rf'{os.getcwd()}\trainingData.csv')
test_df = pd.read_csv(rf'{os.getcwd()}\testingData2022.csv')

# print(type(dates.datestr2num(test_df[0][DATE_COLUMN_NAME])))

convert_to_applicable_types(train_df)
convert_to_applicable_types(test_df)


plt.plot(test_df[DATE_COLUMN_NAME], test_df[PRICE_COLUMN_NAME])
plt.show()

print(test_df.tail())
