import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from preprocessing import preprocess_images

class CNNModel:
    def __init__(self):
        self.cnn_model = self.build_cnn_model()

    def build_cnn_model(self):
        # Load the VGG16 model pretrained on ImageNet
        base_model = VGG16(weights='imagenet', include_top=False)

        # Freeze the VGG16 weights
        for layer in base_model.layers:
            layer.trainable = False

        # Extract feature maps from the 5th convolutional layer (block5_conv3)
        cnn_output = base_model.get_layer('block5_conv3').output

        # Create the model
        cnn_model = Model(inputs=base_model.input, outputs=cnn_output)

        return cnn_model

    def get_features(self, directory_path, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        flag = True

        for img_name in os.listdir(directory_path):
            img_path = os.path.join(directory_path, img_name)
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            features = self.cnn_model.predict(img_array)  # Shape (1, 14, 14, 512)
            features = np.squeeze(features)  # Remove the batch dimension, shape (14, 14, 512)

            if flag:
                print(f'Feature shape: {features.shape}\n')
                input('Press enter to continue...')
                flag = False
            feature_save_path = os.path.join(save_path, f'{os.path.splitext(img_name)[0]}_features.npy')
            np.save(feature_save_path, features)

# Example usage:
cnn_model = CNNModel()
# preprocess_images('F:\\Datasets\\archive\\flickr30k_images\\flickr30k_images\\flickr30k_images', 'F:\\Datasets\\archive\\flickr30k_images\\processed_images')
# cnn_model.get_features('F:\\Datasets\\archive\\flickr30k_images\\processed_images', 'F:\\Datasets\\archive\\flickr30k_images\\features')

preprocess_images('data\\images\\raw', 'data\\images\\processed')
cnn_model.get_features('data\\images\\processed', 'data\\features')