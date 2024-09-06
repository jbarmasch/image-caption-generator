import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from lstm import build_lstm_model
from preprocessing import load_image_features

# Load tokenizer
tokenizer_path = 'data\\tokenizer\\tokenizer.pkl'
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Parameters
max_len = 20
vocab_size = 10000  # This should match the vocab_size used during training
image_path = 'data\\images\\processed'  # Replace with the directory containing new images
features_path = 'data\\features'  # Path to precomputed features

# Initialize and load the trained model
vocab_size = len(tokenizer.word_index) + 1
model = build_lstm_model(vocab_size=vocab_size)
model.load_weights('Training results\\Weights\\LSTM\\saat\\model_weights_02.weights.h5')  # Replace with the path to your trained model weights

def decode_sequence(sequence, tokenizer):
    reverse_word_index = {value: key for key, value in tokenizer.word_index.items()}
    return ' '.join([reverse_word_index.get(idx, '') for idx in sequence if idx > 0])

def predict_caption(features, model, tokenizer, max_len=20, temperature=0.65, top_k=3):
    start_token = tokenizer.word_index.get('startseq', 1)
    end_token = tokenizer.word_index.get('endseq')
    caption_input = np.zeros((1, max_len))
    caption_input[0, 0] = start_token

    predicted_caption = [start_token]

    for i in range(1, max_len):
        preds = model.predict([features, caption_input], verbose=0)[0, i-1, :]

        # Apply temperature scaling
        preds = np.log(preds + 1e-2) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        # Top-K sampling
        top_k_indices = np.argsort(preds)[-top_k:]  # Get indices of the top-K predictions
        top_k_probs = preds[top_k_indices]  # Probabilities of top-K words
        top_k_probs /= np.sum(top_k_probs)  # Normalize probabilities

        # Sample from top-K
        next_word_idx = np.random.choice(top_k_indices, p=top_k_probs)

        # Stop if the end token is predicted or max length is reached
        if next_word_idx == 0 or (end_token is not None and next_word_idx == end_token):
            break

        caption_input[0, i] = next_word_idx
        predicted_caption.append(next_word_idx)
    
    return decode_sequence(predicted_caption[1:], tokenizer)  # Skip the start token

def plot_image_with_caption(image_name, caption, image_path):
    img_full_path = os.path.join(image_path, f'{image_name}.jpg')
    img = load_img(img_full_path, target_size=(224, 224))
    plt.imshow(img)
    plt.title(caption)
    plt.axis('off')
    plt.show()

# Predict captions for new images
for img_name in os.listdir(image_path):
    image_name = os.path.splitext(img_name)[0]  # Remove file extension to get base name
    features = load_image_features(image_name, features_path).reshape(1, 196, 512)
    predicted_caption = predict_caption(features, model, tokenizer, max_len)
    plot_image_with_caption(image_name, predicted_caption, image_path)