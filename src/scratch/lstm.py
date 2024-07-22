from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense

def build_lstm(vocab_size, max_sequence_length, embedding_dim=256):
    text_input = Input(shape=(max_sequence_length,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(text_input)
    x = LSTM(512, return_sequences=True)(x)
    lstm_model = Model(inputs=text_input, outputs=x)
    return lstm_model
