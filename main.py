from neuralprophet import NeuralProphet
import pandas as pd
from matplotlib import pyplot as plt
import os

PRICE_COLUMN_NAME = 'Value'
DATE_COLUMN_NAME = 'DateTime'


def convert_to_applicable_types(df):
    df[PRICE_COLUMN_NAME] = pd.to_numeric(df[PRICE_COLUMN_NAME])
    df[DATE_COLUMN_NAME] = pd.to_datetime(df[DATE_COLUMN_NAME])


def reformat_for_prophet():
    convert_to_applicable_types(train_df)
    convert_to_applicable_types(test_df)
    train_df.columns = ['ds', 'y']
    train_df.ds = train_df.ds.drop_duplicates()
    train_df.dropna(inplace=True)
    test_df.columns = ['ds', 'y']
    test_df.ds = test_df.ds.drop_duplicates()
    test_df.dropna(inplace=True)


train_df = pd.read_csv(rf'{os.getcwd()}\trainingData.csv')
test_df = pd.read_csv(rf'{os.getcwd()}\testData.csv')

reformat_for_prophet()

m = NeuralProphet()
train_metrics = m.fit(train_df, freq='D')
test_metrics = m.test(test_df)
print(f'============ TEST RESULTS ============\n\n{test_metrics}\n\n====================================\n\n')

forecast = m.predict(test_df)
print(forecast)

plot = m.plot(forecast)
plt.ylabel('MWh')
plt.xlabel('Day')
plot.show()
