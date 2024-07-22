# Dataset iterators
# import flickr30k


# Datasets: 'name', 'load_data: returns iterator', 'prepare_data: some preprocessing'
# datasets = { #'flickr8k': (flickr8k.load_data, flickr8k.prepare_data),
#            # 'coco': (coco.load_data, coco.prepare_data),
#             'flickr30k': (flickr30k.load_data, flickr30k.prepare_data),
#             }

# def get_dataset(name):
#    return datasets[name][0], datasets[name][1]


import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import numpy as np
import os
from sklearn.model_selection import train_test_split

def preprocess_images(input_directory, output_directory):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        for img_name in os.listdir(input_directory):
            img_path = os.path.join(input_directory, img_name)
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)

            preprocessed_img_path = os.path.join(output_directory, img_name)
            save_img(preprocessed_img_path, img_array)

def load_captions(captions_path):
    # Load the captions CSV file
    captions_df = pd.read_csv(captions_path, header=None, names=['image_name', 'comment_number', 'comment'], delimiter='|')
    return captions_df

def preprocess_captions(captions_df, max_len=20):
    # Extract the captions from the dataframe
    captions = captions_df['comment'].tolist()

    # Ensure all captions are strings
    captions = [str(caption) for caption in captions]

    # Create a tokenizer and fit it on the captions
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(captions)

    return tokenizer

def load_image_features(image_name, features_path):
    features_file = os.path.join(features_path, f"{image_name}_features.npy")
    features = np.load(features_file)
    features = np.reshape(features, (196, 512))
    return features

def create_data_generator(captions_df, tokenizer, features_path, max_len=20, batch_size=64, vocab_size=10000):

    while True:
            for start in range(0, len(captions_df), batch_size):
                end = min(start + batch_size, len(captions_df))
                batch_df = captions_df.iloc[start:end]
                
                batch_images = []
                batch_sequences = []
                batch_labels = []
                
                for idx, row in batch_df.iterrows():
                    image_name = row['image_name'].replace('.jpg', '')
                    features = np.load(f'{features_path}/{image_name}_features.npy')
                    features = features.reshape((196, 512))
                    
                    caption = str(row['comment'])
                    seq = tokenizer.texts_to_sequences([caption])[0]
                    seq = pad_sequences([seq], maxlen=max_len, padding='post')[0]
                    
                    batch_images.append(features)
                    batch_sequences.append(seq)
                    batch_labels.append(tf.keras.utils.to_categorical(seq, num_classes=vocab_size))
                
                batch_images = np.array(batch_images)
                batch_sequences = np.array(batch_sequences)
                batch_labels = np.array(batch_labels)
                
                yield ((batch_images, batch_sequences), batch_labels)

def create_tf_data_generator(captions_df, tokenizer, features_path, max_len, batch_size, vocab_size):
    output_signature = (
        (
            tf.TensorSpec(shape=(None, 196, 512), dtype=tf.float32),
            tf.TensorSpec(shape=(None, max_len), dtype=tf.int32)
        ),
        tf.TensorSpec(shape=(None, max_len, vocab_size), dtype=tf.float32)
    )
    
    generator = create_data_generator(captions_df, tokenizer, features_path, max_len, batch_size, vocab_size)
    
    dataset = tf.data.Dataset.from_generator(generator)

    return dataset

def create_datasets(captions_df, test_size=0.2):
    return train_test_split(
        captions_df, test_size=test_size, random_state=42
    )