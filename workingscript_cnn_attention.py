import os
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Dropout
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

import cv2
import random
from scipy.ndimage import rotate

# Function to apply data augmentation (rotation, erosion, dilation) to training images
def augment_images_and_labels(images, labels, augment_factor=1):
    augmented_images, augmented_labels = [], []

    for _ in range(augment_factor):
        for image, label in zip(images, labels):
            # Randomly apply rotation
            if random.random() < 0.5:
                angle = random.uniform(-15, 15)  # Rotate between -15 and +15 degrees
                image_aug = rotate(image, angle, reshape=False, mode='nearest')
            else:
                image_aug = image.copy()

            # Randomly apply erosion
            if random.random() < 0.5:
                kernel = np.ones((2, 2), np.uint8)
                image_aug = cv2.erode(image_aug, kernel, iterations=1)

            # Randomly apply dilation
            if random.random() < 0.5:
                kernel = np.ones((2, 2), np.uint8)
                image_aug = cv2.dilate(image_aug, kernel, iterations=1)

            # Resize the augmented image to maintain a consistent shape (32, 128)
            image_aug = cv2.resize(image_aug, (128, 32))

            # Add a channel dimension to each augmented image to match original images' shape
            image_aug = np.expand_dims(image_aug, axis=-1)

            # Add the augmented image and its corresponding label
            augmented_images.append(image_aug)
            augmented_labels.append(label)

    return np.array(augmented_images), augmented_labels

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, Add, Flatten,
    RepeatVector, Bidirectional, LSTM, Dense, TimeDistributed, Layer, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.backend import ctc_batch_cost
from keras import backend as K
# Residual block definition
def residual_block(x, filters):
    """Residual Block with two Conv2D layers and a skip connection."""
    residual = Conv2D(filters, (1, 1), padding='same')(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
 #   x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
#    x = BatchNormalization()(x)
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
import tensorflow as tf
import tensorflow.keras.backend as K

def ctc_loss(y_true, y_pred):
    # Compute the lengths of the sequences (y_pred)
    input_length = tf.ones(shape=(tf.shape(y_pred)[0], 1), dtype=tf.float32) * tf.cast(tf.shape(y_pred)[1], dtype=tf.float32)
    input_length = tf.cast(input_length, dtype=tf.int32)  # Convert to int32 after multiplication

    # Compute the label lengths based on non-zero elements in y_true
    label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), dtype=tf.int32), axis=1)  # Remove axis=-1 to avoid squeezing issues

    # Calculate the CTC loss
    loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# Build CNN-LSTM model with residual blocks, attention, and CTC loss
# Build CNN-LSTM model with residual blocks, attention, and CTC loss
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, 
    Attention, TimeDistributed, LayerNormalization, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_cnn_attention(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # CNN Layers: Feature Extraction
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Flatten CNN output and reshape for Attention input
    x = Flatten()(x)  # Shape: (None, features)
    x = Dense(116 * 256)(x)  # Ensure it matches the sequence length
    x = Reshape((116, 256))(x)  # Shape: (batch_size, 116, 256)

    # Attention Mechanism: Focus on relevant parts of the sequence
    # First Attention Mechanism
    query_1 = Dense(256)(x)  # Shape: (batch_size, 116, 256)
    key_1 = Dense(256)(x)     # Shape: (batch_size, 116, 256)
    value_1 = Dense(256)(x)   # Shape: (batch_size, 116, 256)

    attention_output_1 = Attention()([query_1, key_1, value_1])  # Shape: (batch_size, 116, 256)

    # Optional: Apply normalization and dropout after first attention layer
    x = LayerNormalization()(attention_output_1)
    x = Dropout(0.2)(x)

    # Second Attention Mechanism
    query_2 = Dense(256)(x)
    key_2 = Dense(256)(x)
    value_2 = Dense(256)(x)

    attention_output_2 = Attention()([query_2, key_2, value_2])  # Shape: (batch_size, 116, 256)

    # Optional: Apply normalization and dropout after second attention layer
    x = LayerNormalization()(attention_output_2)
    x = Dropout(0.2)(x)

    # Map output to required shape (None, 116, 116)
    outputs = TimeDistributed(Dense(116, activation='softmax'))(x)

    # Build the model
    model = Model(inputs, outputs)

    # Compile the model
    optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Main function to call all required functions
def main():
    # Specify paths
    dataset_path = r'/home/springschool_local/Shaukatali/USHWRdataset/UHWR'  # Replace with actual path to the dataset
    chars_file = r'/home/springschool_local/Shaukatali/USHWRdataset/UHWR/chars.txt'  # Replace with actual path to the character file
    chars = load_chars(chars_file)

    # Load data
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_data(dataset_path, chars)

    # Apply data augmentation to training images only and extend labels
    augment_factor = 2  # Adjust this for the number of augmented copies desired
    augmented_train_images, augmented_train_labels = augment_images_and_labels(train_images, train_labels, augment_factor)

    # Combine original and augmented data
    extended_train_images = np.concatenate((train_images, augmented_train_images), axis=0)
    # Ensure labels are lists before concatenating
    extended_train_labels = list(train_labels) + list(augmented_train_labels)
    extended_train_labels = np.array(extended_train_labels)
    print(extended_train_images.shape,extended_train_labels.shape)

    # Define model input shape
    input_shape = (32, 128, 1)  # Height, Width, Channels
    model = build_cnn_attention(input_shape, len(chars))

    # Model summary
    model.summary()

    batch_size = 32
    epochs = 100  # Adjust based on your needs

    # Train the model
    history = model.fit(extended_train_images,extended_train_labels,batch_size=batch_size,epochs=epochs,validation_data=(val_images, val_labels),verbose=1 )

    # Optionally evaluate the model on the test set
    test_loss = model.evaluate(test_images, test_labels, verbose=1)
    print(f'Test Loss: {test_loss}')
if __name__ == "__main__":
    main()
