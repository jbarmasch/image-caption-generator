import tensorflow as tf
from tensorflow.keras import layers, Model

class CNN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        # Define the input layer
        input_layer = layers.Input(shape=self.input_shape)

        # Add convolutional layers
        conv_layer1 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
        pool_layer1 = layers.MaxPooling2D(pool_size=(2, 2))(conv_layer1)
        
        conv_layer2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool_layer1)
        pool_layer2 = layers.MaxPooling2D(pool_size=(2, 2))(conv_layer2)
        
        # Flatten the feature maps
        flatten_layer = layers.Flatten()(pool_layer2)
        
        # Add fully connected layers
        fc_layer1 = layers.Dense(512, activation='relu')(flatten_layer)
        output_layer = layers.Dense(self.num_classes, activation='softmax')(fc_layer1)
        
        # Create the model
        model = Model(inputs=input_layer, outputs=output_layer)
        return model

    def train(self, train_dataset, val_dataset, epochs=10, batch_size=32):
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        
        history = self.model.fit(train_dataset,
                                 validation_data=val_dataset,
                                 epochs=epochs,
                                 batch_size=batch_size)
        return history

    def evaluate(self, test_dataset):
        return self.model.evaluate(test_dataset)

    def predict(self, image):
        # Preprocess the image if needed
        # Make predictions using the model
        return self.model.predict(image)

# Example usage:
# Define input shape and number of classes
# input_shape = (224, 224, 3)
# num_classes = 10

# # Create an instance of the CNN model
# cnn_model = CNN(input_shape, num_classes)

# # Print model summary
# cnn_model.model.summary()
