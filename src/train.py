import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from image_preprocessing import preprocess_image, process_directory
from cnn import CNN
from lstm import LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Embedding, Input, Concatenate, Reshape
from nltk.translate.bleu_score import sentence_bleu

# Directory paths
input_directory = "./data/images/raw/"  # Directory containing raw images
output_directory = "./data/images/processed/"  # Directory to save preprocessed images
captions_file = "./data/annotations/captions.csv"  # Path to captions file
model_save_path = "./models/image_caption_model.h5"  # Path to save the trained model

# Parameters
target_size = (224, 224)  # Target size for image resizing
batch_size = 32
epochs = 10
max_length = 34  # Maximum length of captions (adjust as necessary)
vocab_size = 5000  # Maximum number of words in tokenizer (adjust as necessary)
embedding_dim = 256

def load_captions(captions_file):
    """Load captions from CSV file."""
    captions = pd.read_csv(captions_file, delimiter='|', header=None, names=['image_name', 'comment_number', 'comment'])
    return captions

def tokenize_captions(captions, vocab_size):
    """Tokenize and pad captions."""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts(captions)
    sequences = tokenizer.texts_to_sequences(captions)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    return padded_sequences, tokenizer

class ImageCaptionGenerator(Sequence):
    """Helper to generate batches of image-caption pairs."""

    def __init__(self, image_filenames, captions, batch_size, target_size):
        self.image_filenames = image_filenames
        self.captions = captions
        self.batch_size = batch_size
        self.target_size = target_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.captions[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = [preprocess_image(cv2.imread(os.path.join(output_directory, filename)), self.target_size) for filename in batch_x]

        return [np.array(images), np.array(batch_y)], np.array(batch_y)

def build_combined_model(cnn_model, lstm_model):
    """Combine CNN and LSTM models for image captioning."""
    cnn_output = cnn_model.model.output
    cnn_output = Reshape((1, -1))(cnn_output)
    
    lstm_input = Input(shape=(max_length,))
    lstm_embedding = Embedding(vocab_size, embedding_dim, input_length=max_length)(lstm_input)
    
    combined_input = Concatenate()([cnn_output, lstm_embedding])
    lstm_output = lstm_model.model(combined_input)
    
    combined_model = tf.keras.Model(inputs=[cnn_model.model.input, lstm_input], outputs=lstm_output)
    return combined_model

def evaluate_model(model, val_generator, tokenizer, captions_df):
    """Evaluate the model using BLEU score."""
    references = []
    hypotheses = []

    for batch in val_generator:
        (images, input_captions), true_captions = batch
        predictions = model.predict([images, input_captions])

        for i in range(len(predictions)):
            predicted_caption = np.argmax(predictions[i], axis=1)
            true_caption = true_captions[i]

            predicted_caption_text = ' '.join([tokenizer.index_word[word] for word in predicted_caption if word in tokenizer.index_word])
            true_caption_texts = captions_df[captions_df['image_name'] == val_generator.image_filenames[i]]['comment'].tolist()

            references.append([true_caption_text.split() for true_caption_text in true_caption_texts])
            hypotheses.append(predicted_caption_text.split())

    bleu_scores = [sentence_bleu(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    average_bleu = np.mean(bleu_scores)
    return average_bleu

def main():
    # Preprocess and store images
    process_directory(input_directory, output_directory, target_size)
    
    # Load captions
    captions_df = load_captions(captions_file)
    
    # Create a list of all image paths and corresponding captions
    all_images = []
    all_captions = []
    
    for filename in os.listdir(output_directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_captions = captions_df[captions_df['image_name'] == filename]
            for _, row in image_captions.iterrows():
                all_images.append(filename)
                all_captions.append(row['comment'])
    
    # Tokenize captions
    padded_sequences, tokenizer = tokenize_captions(all_captions, vocab_size)
    
    # Split data into training and validation sets
    train_images, val_images, train_captions, val_captions = train_test_split(
        all_images, padded_sequences, test_size=0.2, random_state=42)
    
    # Build models
    input_shape_cnn = (224, 224, 3)
    input_shape_lstm = (max_length,)
    num_classes = vocab_size
    
    cnn_model = CNN(input_shape_cnn, num_classes)
    lstm_model = LSTM(input_shape_lstm, num_classes, num_lstm_layers=2, lstm_units=512)
    
    # Combine models
    combined_model = build_combined_model(cnn_model, lstm_model)
    combined_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Prepare datasets
    train_generator = ImageCaptionGenerator(train_images, train_captions, batch_size, target_size)
    val_generator = ImageCaptionGenerator(val_images, val_captions, batch_size, target_size)
    
    # Train model
    combined_model.fit(train_generator, validation_data=val_generator, epochs=epochs)
    
    # Evaluate model
    average_bleu = evaluate_model(combined_model, val_generator, tokenizer, captions_df)
    print(f"Validation BLEU score: {average_bleu}")
    
    # Save model
    combined_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
