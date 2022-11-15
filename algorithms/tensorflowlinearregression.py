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
import utilities
import constants
from utilities import setup_plot_with_data

train_df, test_df = utilities.get_data()

setup_plot_with_data(train_df, test_df)
plt.plot(test_df[constants.DATE_COLUMN_NAME], test_df[constants.PRICE_COLUMN_NAME])
plt.show()
