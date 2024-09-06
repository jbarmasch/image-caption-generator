import os
import numpy as np
import tensorflow as tf
from lstm import build_lstm_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from keras.models import load_model
from preprocessing import load_image_features, preprocess_captions, load_captions

# Set paths
features_path = 'data\\features'
image_path = 'data\\images\\processed'
model_weights_path = 'Training results\\Weights\\LSTM\\saat\\model_weights_03.weights.h5'
tokenizer_path = 'data\\tokenizer\\tokenizer.pkl'

# Load tokenizer
import pickle
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define max length of captions
max_len = 20

# Build the LSTM model (assuming the same architecture as used for training)
vocab_size = len(tokenizer.word_index) + 1

model = build_lstm_model(vocab_size)
model.load_weights(model_weights_path)

# Function to generate a caption for an image
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.split()[1:-1]  # Remove 'startseq' and 'endseq'
    return ' '.join(final_caption)

# Load and preprocess images, generate captions
for image_name in os.listdir(image_path):
    if image_name.endswith('.JPG'):
        image_id = image_name.split('.')[0]
        features = load_image_features(image_id, features_path).reshape(1, 196, 512)
        print(f"Generating caption for image: {image_name}")
        caption = generate_caption(model, tokenizer, features, max_len)
        print(f'Image: {image_name} Caption: {caption}')