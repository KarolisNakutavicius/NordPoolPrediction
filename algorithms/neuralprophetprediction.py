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
train_df, test_df = utilities.get_data()
utilities.setup_plot_with_data(train_df, test_df)
reformat_for_prophet()

# Train
m = NeuralProphet()
train_metrics = m.fit(train_df, freq='D')

# Test
test_metrics = m.test(test_df)
print(f'============ TEST RESULTS ============\n\n{test_metrics}\n\n====================================\n\n')

# Predict
forecast = m.predict(test_df)
print(forecast)

plot = m.plot(forecast)
plot.show()
