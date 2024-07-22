from preprocessing import load_captions, preprocess_captions
from lstm import build_lstm_model
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

def plot_image_with_caption(img_path, caption):
    img = image.load_img(img_path, target_size=(224, 224))
    plt.imshow(img)
    plt.title(caption)
    plt.axis('off')
    plt.show()

def generate_caption(model, tokenizer, image_features, max_len=20):
    start_token = tokenizer.word_index['<start>']
    end_token = tokenizer.word_index['<end>']
    
    caption_seq = np.zeros((1, max_len))
    caption_seq[0, 0] = start_token
    
    for i in range(1, max_len):
        predictions = model.predict([image_features, caption_seq], verbose=0)
        predicted_id = np.argmax(predictions[0, i-1])
        
        caption_seq[0, i] = predicted_id
        
        if predicted_id == end_token:
            break
    
    caption = decode_sequence(caption_seq[0], tokenizer)
    return caption

def decode_sequence(sequence, tokenizer):
    reverse_word_index = {value: key for key, value in tokenizer.word_index.items()}
    decoded_words = []
    for idx in sequence:
        if idx == 0:
            continue
        word = reverse_word_index.get(idx, '')
        if word == '<end>':
            break
        decoded_words.append(word)
    return ' '.join(decoded_words)

def main(images_path, features_path, model_weights, tokenizer_path, max_len=20):
    
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    
    # Build the model and load weights
    vocab_size = len(tokenizer.word_index) + 1
    model = build_lstm_model(vocab_size=vocab_size)
    model.load_weights(model_weights)
    model.summary()
    
    # Get list of image files
    image_files = [f for f in os.listdir(images_path) if f.endswith('.JPG')]
    
    for img_file in image_files:
        img_path = os.path.join(images_path, img_file)
        feature_file = os.path.join(features_path, img_file.replace('.JPG', '_features.npy'))
        
        if os.path.exists(feature_file):
            # Load precomputed features
            image_features = np.load(feature_file)
            image_features = image_features.reshape((1, 196, 512))  # Adjust shape as needed
            
            # Generate caption
            caption = generate_caption(model, tokenizer, image_features, max_len=max_len)
            
            # Plot image with caption
            plot_image_with_caption(img_path, caption)

if __name__ == "__main__":
    features_path = 'data\\features'
    images_path = 'data\\images\\processed'
    model_weights_path = 'Training results\\Weights\\LSTM\\saat\\model_weights_28.weights.h5'
    tokenizer_path = 'data\\tokenizer\\tokenizer.pkl'
    
    main(images_path, features_path, model_weights_path, tokenizer_path)