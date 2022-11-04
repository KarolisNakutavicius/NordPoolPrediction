from neuralprophet import NeuralProphet
import pandas as pd
from matplotlib import pyplot as plt

PRICE_COLUMN_NAME = 'Value'
DATE_COLUMN_NAME = 'Date'


def convert_to_applicable_types():
    df[PRICE_COLUMN_NAME] = df[PRICE_COLUMN_NAME].str.replace(',', '.')
    df[PRICE_COLUMN_NAME] = pd.to_numeric(df[PRICE_COLUMN_NAME])
    df[DATE_COLUMN_NAME] = pd.to_datetime(df[DATE_COLUMN_NAME], dayfirst=True)
    df[DATE_COLUMN_NAME] = df[DATE_COLUMN_NAME].apply(lambda x: x.Year)


df = pd.read_csv(r'C:\Users\karol\Desktop\Dienos-kainos.csv')
convert_to_applicable_types()

plt.plot(df[DATE_COLUMN_NAME], df[PRICE_COLUMN_NAME])
plt.title("Electricity price")
plt.ylabel('MWh')
plt.xlabel('Day')
plt.tick_params(axis='x', which='major', labelsize=5)

plt.show()
