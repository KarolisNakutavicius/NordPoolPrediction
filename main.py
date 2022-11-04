from neuralprophet import NeuralProphet
import pandas as pd
from matplotlib import pyplot as plt


def convert_to_applicable_types():
    df['Value'] = df['Value'].str.replace(',', '.')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Value'] = pd.to_numeric(df['Value'])


df = pd.read_csv(r'C:\Users\karol\Desktop\Dienos-kainos.csv')
convert_to_applicable_types()

print(df.dtypes)
