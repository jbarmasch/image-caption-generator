import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Embedding, LSTM, add
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Define the CNN for feature extraction
def create_cnn(input_shape):
    cnn = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),  # Output size for feature vectors
    ])
    return cnn

# Define the RNN for caption generation
def create_rnn(vocab_size, max_caption_length):
    rnn = Sequential([
        Embedding(vocab_size, 256, input_length=max_caption_length),
        LSTM(256, return_sequences=True),
        LSTM(256),
        Dense(256, activation='relu'),
    ])
    return rnn

# Combine CNN and RNN
def create_model(input_shape, vocab_size, max_caption_length):
    cnn = create_cnn(input_shape)
    rnn = create_rnn(vocab_size, max_caption_length)
    
    # Define inputs
    image_input = tf.keras.Input(shape=input_shape)
    caption_input = tf.keras.Input(shape=(max_caption_length,))
    
    # Extract features from the image using the CNN
    image_features = cnn(image_input)
    
    # Generate caption features using the RNN
    caption_features = rnn(caption_input)
    
    # Combine image and caption features
    combined = add([image_features, caption_features])
    output = Dense(vocab_size, activation='softmax')(combined)
    
    model = Model(inputs=[image_input, caption_input], outputs=output)
    return model

input_shape = (224, 224, 3)  # Example input shape for the CNN
vocab_size = 10000  # Example vocabulary size
max_caption_length = 16  # Example max length of a caption

model = create_model(input_shape, vocab_size, max_caption_length)
model.summary()


#######################################################################################
#######################################################################################
#######################################################################################



# Load and preprocess the dataset
def preprocess_caption(caption, tokenizer, max_caption_length):
    caption_seq = tokenizer.texts_to_sequences([caption])[0]
    caption_seq = pad_sequences([caption_seq], maxlen=max_caption_length, padding='post')[0]
    return caption_seq

# Tokenize the captions
tokenizer = Tokenizer(num_words=vocab_size)
all_captions = [caption for image_id in image_id_to_captions for caption in image_id_to_captions[image_id]]
tokenizer.fit_on_texts(all_captions)

# Example generator function to yield image and caption pairs
def data_generator(images_path, annotations, tokenizer, max_caption_length, batch_size):
    while True:
        image_batch = []
        caption_batch = []
        for image_info in annotations['images']:
            image_id = image_info['id']
            image_file = os.path.join(images_path, image_info['file_name'])
            if os.path.exists(image_file):
                image = load_and_preprocess_image(image_file)
                captions = image_id_to_captions.get(image_id, [])
                for caption in captions:
                    caption_seq = preprocess_caption(caption, tokenizer, max_caption_length)
                    image_batch.append(image)
                    caption_batch.append(caption_seq)
                    if len(image_batch) == batch_size:
                        yield [np.array(image_batch), np.array(caption_batch)], np.array(caption_batch)
                        image_batch, caption_batch = [], []

train_generator = data_generator(train_images_path, annotations, tokenizer, max_caption_length, batch_size=32)

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(train_generator, epochs=10, steps_per_epoch=1000)
