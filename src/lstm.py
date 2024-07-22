import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm(vocab_size, embedding_dim=256, lstm_units=512):
    model = models.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, mask_zero=True))
    model.add(layers.LSTM(lstm_units, return_sequences=True, dropout=0.5))
    model.add(layers.LSTM(lstm_units, return_sequences=False, dropout=0.5))
    return model

####################################################################################################