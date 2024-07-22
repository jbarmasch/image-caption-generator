import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Add, Dot, Activation, Concatenate
from tensorflow.keras.optimizers import Adam

# Import the LSTM model and the CNN feature extractor
from lstm import build_lstm_model
from cnn import build_cnn_model

def build_combined_model(vocab_size=10000, max_len=20, embed_dim=256, lstm_units=512, dropout_rate=0.5):
    # Build the CNN model to extract image features
    cnn_model = build_cnn_model()
    
    # Inputs for image features (output of CNN) and sequences
    inputs_image = cnn_model.output
    inputs_seq = Input(shape=(max_len,))
    
    # Sequence processing (Embedding + LSTM layers)
    x_seq = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len)(inputs_seq)
    x_seq = LSTM(lstm_units, return_sequences=True)(x_seq)
    x_seq = Dropout(dropout_rate)(x_seq)
    
    # Attention mechanism
    attention = Dot(axes=[2, 2])([x_seq, inputs_image])
    attention = Activation('softmax')(attention)
    context = Dot(axes=[2, 1])([attention, inputs_image])
    
    # Combine context and sequence output
    combined = Concatenate()([context, x_seq])
    outputs = Dense(vocab_size, activation='softmax')(combined)
    
    # Create and compile model
    model = Model(inputs=[cnn_model.input, inputs_seq], outputs=outputs)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, data, batch_size=64, epochs=10):
    # Assuming 'data' contains (train_images, train_sequences, train_labels)
    train_images, train_sequences, train_labels = data

    # Fit the model
    model.fit([train_images, train_sequences], train_labels, batch_size=batch_size, epochs=epochs)


