import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import re
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers
from keras.applications import efficientnet
from keras.layers import TextVectorization
from keras.callbacks import ModelCheckpoint


keras.utils.set_random_seed(111)

from preprocessing import train_dataset, valid_dataset, vectorization, train_data, valid_data, text_data, decode_and_resize
from model import build_combined_model

# Path to the images
IMAGES_PATH = "F:\\Datasets\\8k\\Images"

# Desired image dimensions
IMAGE_SIZE = (299, 299)

INPUT_SHAPE = (224, 224, 3)

# Vocabulary size
VOCAB_SIZE = 10000

# Fixed length allowed for any sequence
SEQ_LENGTH = 30

# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512

# Per-layer units in the feed-forward network
FF_DIM = 512

# Other training parameters
BATCH_SIZE = 64
EPOCHS = 350
AUTOTUNE = tf.data.AUTOTUNE

# EarlyStopping criteria
early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Create a learning rate schedule
num_train_steps = len(train_dataset) * EPOCHS
num_warmup_steps = num_train_steps // 15

# Compile the model
model = build_combined_model(INPUT_SHAPE, VOCAB_SIZE, SEQ_LENGTH)
print(model.summary())

# Callback for saving model weights
checkpoint = ModelCheckpoint(
    filepath="Training results\\Weights\\LSTM\\1e-5\\model_weights_{epoch:02d}.weights.h5",
    save_weights_only=True,
    save_freq="epoch",
)

print("Train_data keys: ", train_data.keys().__len__())
print("Train_data values: ", train_data.values().__len__())
print("Valid_data keys: ", valid_data.keys().__len__())
print("Valid_data values: ", valid_data.values().__len__())
print("Tesxt data length: ", text_data.__len__())

# Fit the model
history = model.fit(
    [np.array(train_dataset), np.array(text_data)],
    epochs=EPOCHS,
    validation_data=[np.array(valid_dataset), np.array(text_data)],
    callbacks=[early_stopping, checkpoint],
)


### Check Sample predictions ###

vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 1
valid_images = list(valid_data.keys())


def generate_caption():
    # Select a random image from the validation dataset
    sample_img = np.random.choice(valid_images)

    # Read the image from the disk
    sample_img = decode_and_resize(sample_img)
    img = sample_img.numpy().clip(0, 255).astype(np.uint8)
    plt.imshow(img)
    plt.show()

    # Pass the image to the CNN
    img = tf.expand_dims(sample_img, 0)
    img = build_combined_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = build_combined_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = build_combined_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    print("Predicted Caption: ", decoded_caption)

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs", fontsize=14)
plt.legend()
plt.grid()
plt.show()
plt.savefig("Training results\\Graphs\\Loss")

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train and Validation Accuracy Over Epochs")
plt.legend()
plt.grid()
plt.show()
plt.savefig("Training results\\Graphs\\Acc")

# Check predictions for a few samples
generate_caption()
generate_caption()
generate_caption()