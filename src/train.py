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
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.layers import Embedding, Input, Concatenate, Layer, Reshape, Dense, LSTM as KerasLSTM
from nltk.translate.bleu_score import sentence_bleu

# Directory paths
input_directory = "F:\\Datasets\\archive\\flickr30k_images\\flickr30k_images"  # Directory containing raw images
output_directory = ".\\data\\images\\processed\\"  # Directory to save preprocessed images
captions_file = "F:\\Datasets\\archive\\flickr30k_images\\results.csv"  # Path to captions file
model_save_path = ".\\models\\image_caption_model.h5"  # Path to save the trained model

# Parameters
target_size = (224, 224)  # Target size for image resizing
batch_size = 32
epochs = 10
max_length = 36  # Maximum length of captions (adjust as necessary)
vocab_size = 5000  # Maximum number of words in tokenizer (adjust as necessary)
embedding_dim = 256
lstm_units = 512
num_lstm_layers = 5

def load_captions(captions_file):
    """Load captions from CSV file."""
    captions = pd.read_csv(captions_file, delimiter='|', header=None, names=['image_name', 'comment_number', 'comment'])
    return captions

def tokenize_captions(captions, vocab_size):
    """Tokenize and pad captions."""
    captions = [str(caption) for caption in captions]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts(captions)
    sequences = tokenizer.texts_to_sequences(captions)
    padded_sequences = pad_sequences(sequences, maxlen=max_length + 1, padding='post')

    return padded_sequences, tokenizer

class ImageCaptionGenerator(Sequence):
    """Helper to generate batches of image-caption pairs."""

    def __init__(self, image_filenames, captions, batch_size, target_size, vocab_size):
        self.image_filenames = image_filenames
        self.captions = captions
        self.batch_size = batch_size
        self.target_size = target_size
        self.vocab_size = vocab_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.captions[idx * self.batch_size:(idx + 1) * self.batch_size]
    
        images = [preprocess_image(cv2.imread(os.path.join(output_directory, filename)), self.target_size) for filename in batch_x]
    
        # One-hot encode each list of integers in batch_y
        batch_y_one_hot = [to_categorical(np.array(y), num_classes=self.vocab_size) for y in batch_y]
    
        return [np.array(images), np.array(batch_y)], np.array(batch_y_one_hot)

    def on_epoch_end(self):
        pass

def create_dataset(image_filenames, captions, batch_size, target_size, vocab_size):
    output_signature = (
        (
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(35), dtype=tf.int32),
        ),
        tf.TensorSpec(shape=(35, vocab_size), dtype=tf.int32),
    )

    def generator_fn():
        generator = ImageCaptionGenerator(image_filenames, captions, batch_size, target_size, vocab_size)
        for item in generator:
            yield item

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_signature=output_signature,
    )

    return dataset.batch(batch_size)

def build_combined_model(cnn_model, lstm_model, vocab_size, embedding_dim, max_length):
    """Combine CNN and LSTM models for image captioning."""
    # CNN output
    cnn_output = cnn_model.model.output  # Shape: (None, feature_dim)
    cnn_output = Dense(embedding_dim)(cnn_output)  # Reduce to (None, embedding_dim)
    cnn_output = Reshape((1, embedding_dim))(cnn_output)  # Shape: (None, 1, embedding_dim)

    # LSTM input
    lstm_input = Input(shape=(max_length,))
    lstm_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=34)(lstm_input)  # Shape: (None, max_length, embedding_dim)

    # Combine CNN output and LSTM input
    combined_input = Concatenate(axis=1)([cnn_output, lstm_embedding])  # Shape: (None, 1 + max_length, embedding_dim)

    # Define LSTM layers
    lstm_out = lstm_model.model(combined_input)  # Shape: (None, 1 + max_length, lstm_units)

    # Output layer
    output = Dense(vocab_size, activation='softmax')(lstm_out)  # Shape: (None, 1 + max_length, vocab_size)

    combined_model = tf.keras.Model(inputs=[cnn_model.model.input, lstm_input], outputs=output)
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

def preprocess_caption(caption, tokenizer, max_length):
    """Preprocess caption for training."""
    caption = str(caption)
    caption = caption.lower()
    caption = caption.replace('.', '')
    caption = caption.split()
    caption = ['<start>'] + caption + ['<end>']
    caption = ' '.join(caption)
    caption = tokenizer.texts_to_sequences([caption])[0]
    caption = pad_sequences([caption], maxlen=max_length, padding='post')[0]
    return caption


def main():
    # Check if processed images directory is empty
    if not os.listdir(output_directory):
        print("Processed images directory is empty. Processing images...")
        process_directory(input_directory, output_directory, target_size)
    print("Images processed\n")

    # Load captions
    print("Loading captions...")
    captions_df = load_captions(captions_file)
    print("Captions loaded\n")

    # Create a list of all image paths and corresponding captions if does not exist
    print("Checking for image-caption pairs...")
    if not os.path.exists("image_caption_pairs.csv"):
        print("Creating image-caption pairs...")
        all_images = []
        all_captions = []
        
        for filename in os.listdir(output_directory):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_captions = captions_df[captions_df['image_name'] == filename]
                for _, row in image_captions.iterrows():
                    all_images.append(filename)
                    all_captions.append(row['comment'])
        print("Image-caption pairs created\n")

        # Save image-caption pairs to a CSV file
        print("Saving image-caption pairs to CSV file...")
        image_caption_df = pd.DataFrame({'image_name': all_images, 'caption': all_captions})
        image_caption_df.to_csv("image_caption_pairs.csv", index=False)
        print("Image-caption pairs saved to image_caption_pairs.csv\n")
    else:
        # Load image-caption pairs from CSV file
        print("Loading image-caption pairs from CSV file...")
        image_caption_df = pd.read_csv("image_caption_pairs.csv")
        all_images = image_caption_df['image_name'].tolist()
        all_captions = image_caption_df['caption'].tolist()
        print("Image-caption pairs loaded\n")
    
    # Tokenize captions
    print("Tokenizing captions...")
    padded_sequences, tokenizer = tokenize_captions(all_captions, vocab_size)
    print("Captions tokenized\n")
    
    # Split data into training and validation sets
    print("Splitting data into training and validation sets...")
    train_images, val_images, train_captions, val_captions = train_test_split(
        all_images, padded_sequences, test_size=0.2, random_state=42)
    print("Data split\n")

    # Example of preprocessing captions
    print("Preprocessing captions...")
    train_captions = [preprocess_caption(caption, tokenizer, max_length + 1) for caption in train_captions]
    val_captions = [preprocess_caption(caption, tokenizer, max_length + 1) for caption in val_captions]
    print("Captions preprocessed\n")

    # Build models
    # Build CNN model
    print("Building CNN model...")
    input_shape_cnn = (224, 224, 3)
    num_classes = vocab_size
    cnn_model = CNN(input_shape_cnn, num_classes)
    print("CNN model built\n")

    # Build LSTM model
    print("Building LSTM model...")
    input_shape_lstm = (max_length + 1, embedding_dim)
    lstm_model = LSTM(input_shape_lstm, vocab_size, num_lstm_layers=num_lstm_layers, lstm_units=lstm_units)
    print("LSTM model built\n")
    
    # Combine models
    print("Building combined model...")
    combined_model = build_combined_model(cnn_model, lstm_model, vocab_size, embedding_dim, max_length)
    combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Combined model built\n")

    combined_model.summary()

    # Prepare datasets
    print("Preparing datasets...")
    train_generator = ImageCaptionGenerator(train_images, train_captions, batch_size, target_size, vocab_size)
    val_generator = ImageCaptionGenerator(val_images, val_captions, batch_size, target_size, vocab_size)
    
    train_dataset = create_dataset(train_images, train_captions, batch_size, target_size, vocab_size)
    val_dataset = create_dataset(val_images, val_captions, batch_size, target_size, vocab_size)
    print("Datasets prepared\n")
    
    # Train model
    print("Training model...")
    print("Target data shape:", train_dataset.element_spec[0][1].shape)
    combined_model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
    print("Model trained\n")
    
    # Evaluate model
    print("Evaluating model...")
    average_bleu = evaluate_model(combined_model, val_generator, tokenizer, captions_df)
    print(f"Validation BLEU score: {average_bleu}")
    print("Model evaluated\n")
    
    # Save model
    print("Saving model...")
    combined_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()