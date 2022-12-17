import pickle
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
import utilities
import constants
from models.data_type import DataType


def reformat_for_prophet():
    for key in data:
        data[key].columns = ['ds', 'y']
        data[key].ds = data[key].ds.drop_duplicates()
        data[key].dropna(inplace=True)


# Init
data = utilities.init_data()
reformat_for_prophet()

# Train
m = NeuralProphet()
train_metrics = m.fit(data[DataType.TRAIN], validation_df=data[DataType.VALIDATION], freq='D')

with open(constants.NEURAL_PROPHET_MODEL_PATH, "wb") as f:
    pickle.dump(m, f)

# LOAD
# with open(constants.NEURAL_PROPHET_MODEL_PATH, "rb") as f:
#     m = pickle.load(f)

# Test
test_metrics = m.test(data[DataType.TEST])
print(f'============ TEST RESULTS ============\n\n{test_metrics}\n\n====================================\n\n')

# Predict
forecast = m.predict(data[DataType.TEST])

plot = m.plot(forecast)
plt.ylabel('MWh')
plt.xlabel('Date')
plot.show()
