import pickle

import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
import utilities


def reformat_for_prophet():
    train_df.columns = ['ds', 'y']
    train_df.ds = train_df.ds.drop_duplicates()
    train_df.dropna(inplace=True)
    test_df.columns = ['ds', 'y']
    test_df.ds = test_df.ds.drop_duplicates()
    test_df.dropna(inplace=True)


# Init
train_df, test_df = utilities.init_data()
reformat_for_prophet()

# Train
m = NeuralProphet()
train_metrics = m.fit(train_df, freq='D')


with open('saved_model.pkl', "wb") as f:
    pickle.dump(m, f) # Save the model

# with open('saved_model.pkl', "rb") as f:
#     m = pickle.load(f)

# Test
test_metrics = m.test(test_df)
print(f'============ TEST RESULTS ============\n\n{test_metrics}\n\n====================================\n\n')

# Predict
forecast = m.predict(test_df)

plot = m.plot(forecast)
plt.ylabel('MWh')
plt.xlabel('Date')
plot.show()
