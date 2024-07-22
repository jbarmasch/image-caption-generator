import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from preprocessing import load_captions, preprocess_captions, create_datasets, create_data_generator, load_image_features
from lstm import build_lstm_model

# Load and preprocess captions
captions_path = 'F:\\Datasets\\archive\\flickr30k_images\\results.csv'
features_path = 'F:\\Datasets\\archive\\flickr30k_images\\features'
image_path = 'F:\\Datasets\\archive\\flickr30k_images\\processed_images'
max_len = 20

captions_df = load_captions(captions_path)
tokenizer = preprocess_captions(captions_df, max_len=max_len)

# Create datasets
captions_train_df, captions_val_df = create_datasets(captions_df, test_size=0.2)

# Define model parameters
vocab_size = len(tokenizer.word_index) + 1

# Create and compile the model
model = build_lstm_model(vocab_size=vocab_size)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.load_weights('Training results\\Weights\\LSTM\\saat\\model_weights_28.weights.h5')

# Display model summary
model.summary()

captions_df = load_captions(captions_path)
tokenizer = preprocess_captions(captions_df, max_len=max_len)

import pickle
# Define the path where you want to save the tokenizer
tokenizer_path = 'data\\tokenizer\\tokenizer.pkl'

# Save the tokenizer to a pickle file
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Create datasets
captions_train_df, captions_val_df = create_datasets(captions_df, test_size=0.2)

# Create data generators
batch_size = 64
train_generator = create_data_generator(captions_train_df, tokenizer, features_path, max_len=max_len, batch_size=batch_size, vocab_size=vocab_size)
val_generator = create_data_generator(captions_val_df, tokenizer, features_path, max_len=max_len, batch_size=batch_size, vocab_size=vocab_size)

# Calculate steps per epoch
steps_per_epoch_train = len(captions_train_df) // batch_size
steps_per_epoch_val = len(captions_val_df) // batch_size

# EarlyStopping criteria
early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Callback for saving model weights
checkpoint = ModelCheckpoint(
    filepath="Training results\\Weights\\LSTM\\saat\\model_weights_{epoch:02d}.weights.h5",
    save_weights_only=True,
    save_freq="epoch",
)

# Train the model
epochs = 100
# history = model.fit(
#    train_generator,
#    steps_per_epoch=steps_per_epoch_train,
#    epochs=epochs,
#    validation_data=val_generator,
#    validation_steps=steps_per_epoch_val,
#    callbacks = [checkpoint, early_stopping]
#)

# Plot training & validation accuracy values
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.savefig('accuracy.png')
# plt.show()

# Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.savefig('loss.png')
# plt.show()

# Function to decode sequences back to text
def decode_sequence(sequence, tokenizer):
    reverse_word_index = {value: key for key, value in tokenizer.word_index.items()}
    return ' '.join([reverse_word_index.get(idx, '') for idx in sequence if idx > 0])

# Function to plot an image with a caption
def plot_image_with_caption(image_features, caption, tokenizer, model, image_name, image_path):
    # Generate the prediction
    prediction = model.predict([image_features, caption], verbose=0)
    predicted_sequence = np.argmax(prediction[0], axis=1)
    predicted_caption = decode_sequence(predicted_sequence, tokenizer)
    
    # Load and display the image
    img_path = os.path.join(image_path, f'{image_name}.jpg')
    img = image.load_img(img_path, target_size=(224, 224))
    plt.imshow(img)
    plt.title(predicted_caption)
    plt.axis('off')
    plt.show()



# Select a few samples to display
num_samples_to_display = 5
sample_indices = np.random.choice(len(captions_val_df), num_samples_to_display, replace=False)

# Plot images with their predicted captions
for idx in sample_indices:
    image_name = captions_val_df.iloc[idx]['image_name'].replace('.jpg', '')
    features = load_image_features(image_name, features_path).reshape(1, 196, 512)
    caption = pad_sequences(tokenizer.texts_to_sequences([captions_val_df.iloc[idx]['comment']]), maxlen=max_len, padding='post')
    plot_image_with_caption(features, caption, tokenizer, model, image_name, image_path)

###############################################################################