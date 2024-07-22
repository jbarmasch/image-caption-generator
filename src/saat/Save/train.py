from preprocessing import get_dataset
from model import build_combined_model

# Define model parameters
dataset = 'flickr30k'
vocab_size = 10000  # Vocabulary size
max_len = 20  # Max length of sequences
embed_dim = 256
lstm_units = 512
dropout_rate = 0.5
batch_size = 64
epochs = 300

# Load data
load_data, prepare_data = get_dataset(dataset)
train, valid, test, worddict = load_data()

# Build and train the model
model = build_combined_model(vocab_size, max_len, embed_dim, lstm_units, dropout_rate)

history = model.fit([img_train, cap_train, initial_hidden_state], cap_train, 
                        validation_data=([img_val, cap_val, initial_hidden_state[:len(img_val)]], cap_val), 
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping, checkpoint])
    
# Save the model weights
model.save_weights('model_weights.h5')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('loss.png')
plt.show()