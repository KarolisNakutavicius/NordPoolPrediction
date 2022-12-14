import pickle
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
import utilities
import constants
from models.data_type import DataType
from matplotlib.dates import DateFormatter


def reformat_for_prophet():
    for key in data:
        data[key].columns = ['ds', 'y']
        data[key].ds = data[key].ds.drop_duplicates()
        data[key].dropna(inplace=True)


# Init
data = utilities.init_data()
reformat_for_prophet()

# Train
# m = NeuralProphet()
# train_metrics = m.fit(data[DataType.TRAIN], validation_df=data[DataType.VALIDATION], freq='D')

# with open(constants.NEURAL_PROPHET_MODEL_PATH, "wb") as f:
#     pickle.dump(m, f)

# LOAD
with open(constants.NEURAL_PROPHET_MODEL_PATH, "rb") as f:
    m = pickle.load(f)

# Test
test_metrics = m.test(data[DataType.TEST][0:400])
print(f'============ TEST RESULTS ============\n\n{test_metrics}\n\n====================================\n\n')

plot_data = data[DataType.TEST][0:400]

# Predict
forecast = m.predict(plot_data)

plt.plot(plot_data.ds, forecast.yhat1, "-b", label="Prognozuojamos reikšmės")
plt.plot(plot_data.ds, plot_data.y, "-r", label="Tikros reikšmės")
# plt.plot(forecast.ds, "-r", label="Tikros reikšmės")
plt.legend(loc="upper right", fontsize=20)

# plot = m.plot(forecast)
# plot = m.plot_components(forecast)
plt.ylabel('Eur \ MWh')
plt.xlabel('Data')
date_form = DateFormatter("%Y-%m-%d %H:%M")
plt.gca().xaxis.set_major_formatter(date_form)
plt.show()
