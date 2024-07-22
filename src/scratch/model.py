from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, RepeatVector, Concatenate, TimeDistributed, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout

from cnn import build_cnn
from lstm import build_lstm

def build_combined_model(input_shape, vocab_size, max_sequence_length, embedding_dim=256):
    # Build CNN model
    cnn_model = build_cnn(input_shape)
    
    # Build LSTM model
    lstm_model = build_lstm(vocab_size, max_sequence_length, embedding_dim)
    
    # Image feature input
    image_input = cnn_model.input
    image_features = cnn_model.output

    # Caption input
    text_input = lstm_model.input
    lstm_output = lstm_model.output

    # Adjust the shape of image features to match the attention mechanism's input requirement
    repeat_image_features = RepeatVector(max_sequence_length)(image_features)
    
    # Concatenate image features and LSTM output
    concatenated = Concatenate()([lstm_output, repeat_image_features])
    attention = Dense(512, activation='tanh')(concatenated)
    attention = TimeDistributed(Dense(512, activation='relu'))(attention)

    # LSTM to generate caption
    decoder_lstm = LSTM(512, return_sequences=True)(attention)
    decoder_output = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder_lstm)

    # Combined model
    combined_model = Model(inputs=[image_input, text_input], outputs=decoder_output)
    combined_model.compile(optimizer='adam', loss='categorical_crossentropy')

    return combined_model

# Example usage
if __name__ == "__main__":
    input_shape = (224, 224, 3)
    vocab_size = 10000  # example vocabulary size
    max_sequence_length = 20  # example sequence length
    
    model = build_combined_model(input_shape, vocab_size, max_sequence_length)
    model.summary()
