import os
import numpy as np
from PIL import Image

# Function to load images and corresponding labels while ignoring skipped images
def load_images_and_labels(dataset_path, file_list, labels):
    images, valid_labels = [], []
    for img_path, label in zip(file_list, labels):
        full_path = os.path.join(dataset_path, img_path)

        try:
            image = Image.open(full_path).convert('L').resize((128, 32))  # Grayscale and resize
            images.append(np.array(image) / 255.0)  # Normalize
            valid_labels.append(label)  # Only add label if image is successfully loaded
        except FileNotFoundError:
            print(f"Image not found: {full_path}, skipping.")
            continue

    images = np.array(images)
    print(f"Loaded {len(images)} images with shape: {images.shape}")
    return images, valid_labels

# Function to load labels and one-hot encode them
def load_labels(label_list, chars, max_len):
    char_to_index = {char: idx for idx, char in enumerate(chars)}  # Create a dictionary for character to index mapping
    padded_labels = []
    
    for label in label_list:
        # Convert label into one-hot encoding
        one_hot_label = np.zeros((max_len, len(chars)))  # Initialize zero array with shape (max_len, num_chars)
        
        # Fill the label with corresponding one-hot vectors
        for i, char in enumerate(label):
            if char in char_to_index:
                one_hot_label[i][char_to_index[char]] = 1
        
        # Padding with the blank character (index 0 in chars)
        for i in range(len(label), max_len):
            one_hot_label[i][0] = 1  # Set the blank character for padding
        
        padded_labels.append(one_hot_label)

    padded_labels = np.array(padded_labels)
    print(f"Loaded {len(padded_labels)} labels with shape: {padded_labels.shape}")
    return padded_labels

# Function to load both images and labels from text file
def load_file(file_path):
    images_paths, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            img_path, label = line.strip().split('\t')
            images_paths.append(img_path)
            labels.append(label)
    return images_paths, labels

# Find max label length function
def find_max_label_length(label_list):
    return max(len(label) for label in label_list)

# Function to reshape images for CNN-LSTM input
def reshape_for_cnn_lstm(images):
    # Expected input shape: (batch_size, height, width, channels)
    return np.expand_dims(images, axis=-1)  # Adding channel dimension

# Function to load data (separate loading of images and labels)
def load_data(dataset_path, chars):
    train_file = os.path.join(dataset_path, 'train.txt')
    val_file = os.path.join(dataset_path, 'val.txt')
    test_file = os.path.join(dataset_path, 'test.txt')

    # Load file paths and labels
    train_images_paths, train_labels = load_file(train_file)
    val_images_paths, val_labels = load_file(val_file)
    test_images_paths, test_labels = load_file(test_file)

    # Find maximum label length across all datasets (train, val, test)
    max_label_len = max(
        find_max_label_length(train_labels),
        find_max_label_length(val_labels),
        find_max_label_length(test_labels)
    )

    # Load images and labels, ensuring we skip images that are not found
    train_images, train_labels = load_images_and_labels(dataset_path, train_images_paths, train_labels)
    val_images, val_labels = load_images_and_labels(dataset_path, val_images_paths, val_labels)
    test_images, test_labels = load_images_and_labels(dataset_path, test_images_paths, test_labels)

    # Encode the valid labels (after filtering skipped images) to ensure consistency in labels
    train_labels_encoded = load_labels(train_labels, chars, max_label_len)
    val_labels_encoded = load_labels(val_labels, chars, max_label_len)
    test_labels_encoded = load_labels(test_labels, chars, max_label_len)

    # Reshape images for CNN-LSTM input
    train_images = reshape_for_cnn_lstm(train_images)
    val_images = reshape_for_cnn_lstm(val_images)
    test_images = reshape_for_cnn_lstm(test_images)

    return (train_images, train_labels_encoded), (val_images, val_labels_encoded), (test_images, test_labels_encoded)

# Function to load characters
def load_chars(chars_file):
    with open(chars_file, 'r', encoding='utf-8') as f:
        chars = f.read().strip().splitlines()
    chars = [''] + chars  # Add blank character at index 0
    return np.array(chars)


import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, Add, Flatten,
    RepeatVector, Bidirectional, LSTM, Dense, TimeDistributed, Layer, Reshape
)
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.models import Model
from tensorflow.keras.backend import ctc_batch_cost

# Residual block definition
def residual_block(x, filters):
    """Residual Block with two Conv2D layers and a skip connection."""
    residual = Conv2D(filters, (1, 1), padding='same')(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    return Add()([x, residual])

# Attention mechanism layer
class Attention(Layer):
    """Attention mechanism to focus on relevant parts of the sequence."""
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W))
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return output  # Keep the sequence dimension

# CTC loss function
def ctc_loss(y_true, y_pred):
    """CTC loss for handling variable-length sequences."""
    return ctc_batch_cost(y_true, y_pred)

# Build CNN-LSTM model with residual blocks, attention, and CTC loss
def build_cnn_lstm_ctc(input_shape, num_chars, max_label_len):
    inputs = Input(shape=input_shape)

    # CNN with deeper residual blocks
    x = residual_block(inputs, 32)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = residual_block(x, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = residual_block(x, 128)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = residual_block(x, 256)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Reshape instead of flattening for LSTM input
    x = Reshape((-1, x.shape[-1]))(x)  # (batch_size, timesteps, features)

    # LSTM layers with bidirectional wrapping
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = BatchNormalization()(x)

    # Attention layer applied after LSTM
    x = Attention()(x)  # Output shape: (batch_size, timesteps, features)

    # Dynamically adjust the reshape to match the label length
    current_timesteps = tf.shape(x)[1]  # Get the current timestep size dynamically
    x = Reshape((max_label_len, -1))(x[:, :max_label_len, :])  # Pad or truncate

    # Output layer with TimeDistributed for sequence prediction
    outputs = TimeDistributed(Dense(num_chars, activation='softmax'))(x)

    # Build the model
    model = Model(inputs, outputs)

    # Compile with AdamW optimizer and CTC loss
    optimizer = Adam(learning_rate=1e-3, weight_decay=1e-5) 
    model.compile(optimizer=optimizer, loss=ctc_loss)

    return model





# Main function to call all required functions
def main():
    # Specify paths
    dataset_path = r'/home/springschool_local/Shaukatali/USHWRdataset/UHWR'  # Replace with actual path to the dataset
    chars_file = r'/home/springschool_local/Shaukatali/USHWRdataset/UHWR/chars.txt'  # Replace with actual path to the character file
    chars = load_chars(chars_file)

    # Load data
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_data(dataset_path, chars)

    # Print shapes for verification
    print(f"Train images shape: {train_images.shape}, Train labels shape: {train_labels.shape}")
    print(f"Validation images shape: {val_images.shape}, Validation labels shape: {val_labels.shape}")
    print(f"Test images shape: {test_images.shape}, Test labels shape: {test_labels.shape}")
    input_shape = (32, 128, 1)  # height, width, channels
    num_chars = 116  # number of possible characters including blank
    max_label_len = 116  # maximum label length

    # Build and summarize the model
    model = build_cnn_lstm_ctc(input_shape, num_chars, max_label_len)
    model.summary()
    # Train the model for 10 epochs
    history = model.fit(
        train_images, train_labels,  # Training data (images and corresponding one-hot encoded labels)
        epochs=10,  # Train for 10 epochs
        validation_data=(val_images, val_labels),  # Validation data
        batch_size=32  # You can adjust the batch size as needed
    )

    # Evaluate on test data
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc}")

if __name__ == "__main__":
    main()

