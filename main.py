from neuralprophet import NeuralProphet
import pandas as pd
from matplotlib import pyplot as plt

PRICE_COLUMN_NAME = 'Value'
DATE_COLUMN_NAME = 'DateTime'


def convert_to_applicable_types(df):
    df[PRICE_COLUMN_NAME] = pd.to_numeric(df[PRICE_COLUMN_NAME])
    df[DATE_COLUMN_NAME] = pd.to_datetime(df[DATE_COLUMN_NAME])


def show_graph():
    # plt.plot(df[DATE_COLUMN_NAME], df[PRICE_COLUMN_NAME])
    plt.title("Electricity price")
    plt.ylabel('MWh')
    plt.xlabel('Day')
    plt.tick_params(axis='x', which='major', labelsize=8)
    plt.show()


train_df = pd.read_csv(r'C:\Users\karol\Desktop\trainingData.csv')
test_df = pd.read_csv(r'C:\Users\karol\Desktop\testData.csv')
convert_to_applicable_types(train_df)
convert_to_applicable_types(test_df)

# show_graph()

train_df.columns = ['ds', 'y']
train_df.ds = train_df.ds.drop_duplicates()
train_df.dropna(inplace=True)
test_df.columns = ['ds', 'y']
test_df.ds = test_df.ds.drop_duplicates()
test_df.dropna(inplace=True)

m = NeuralProphet()
train_metrics = m.fit(train_df, freq='D')
test_metrics = m.test(test_df)
print(test_metrics)

# future = m.make_future_dataframe(train_df, periods=3)
forecast = m.predict(test_df)
print(forecast)
plot = m.plot(forecast)
# # plot = m.plot_components(forecast)
# show_graph()
plot.show()
