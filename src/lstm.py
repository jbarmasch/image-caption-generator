import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Add, Dot, Activation, RepeatVector, Concatenate
from tensorflow.keras.optimizers import Adam

def build_lstm_model(vocab_size=10000, max_len=20, embed_dim=256, lstm_units=512, dropout_rate=0.5):
    # Input layers
    inputs_image = Input(shape=(196, 512))  # CNN features: 196 vectors of 512 dimensions
    inputs_seq = Input(shape=(max_len,))

    # Image feature extractor (Flattened features from CNN)
    x_image = Dense(embed_dim, activation='relu')(inputs_image)  # Change 512 to embed_dim
    x_image = Dropout(dropout_rate)(x_image)

    # Sequence processing (Embedding + LSTM layers)
    x_seq = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len)(inputs_seq)
    x_seq = LSTM(lstm_units, return_sequences=True)(x_seq)
    x_seq = Dropout(dropout_rate)(x_seq)

    # Project x_seq to the same dimensional space as x_image
    x_seq_proj = Dense(embed_dim, activation='relu')(x_seq)

    # Attention mechanism
    attention = Dot(axes=[2, 2])([x_seq_proj, x_image])  # Align dimensions for dot product
    attention = Activation('softmax')(attention)
    context = Dot(axes=[2, 1])([attention, x_image])

    # Combine context and sequence output
    combined = Concatenate()([context, x_seq])
    outputs = Dense(vocab_size, activation='softmax')(combined)

    # Create and compile model
    model = Model(inputs=[inputs_image, inputs_seq], outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, data, batch_size=64, epochs=10):
    # Assuming 'data' contains (train_images, train_sequences, train_labels)
    train_images, train_sequences, train_labels = data

    # Fit the model
    model.fit([train_images, train_sequences], train_labels, batch_size=batch_size, epochs=epochs)

