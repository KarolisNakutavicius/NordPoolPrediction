import os
WINDOW_SIZE = 5

PRICE_COLUMN_NAME = 'Value'
DATE_COLUMN_NAME = 'DateTime'
TITLE_PREDICTED_PRICES = "Dienos vidutinės kainos"

SAVED_MODELS_PATH = rf'{os.path.abspath(os.path.dirname(__file__))}\results\models'
CNN_MODEL_PATH = rf'{SAVED_MODELS_PATH}\cnn_model'
LSTM_MODEL_PATH = rf'{SAVED_MODELS_PATH}\lstm_model'
GRU_MODEL_PATH = rf'{SAVED_MODELS_PATH}\gru_model'
NEURAL_PROPHET_MODEL_PATH = rf'{SAVED_MODELS_PATH}\neural_prophet\saved_model.pkl'
