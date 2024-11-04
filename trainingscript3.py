import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, BatchNormalization, Flatten, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Load Data Function
def load_data(dataset_path, char_to_index, max_length):
    train_file = os.path.join(dataset_path, 'train.txt')
    val_file = os.path.join(dataset_path, 'val.txt')
    test_file = os.path.join(dataset_path, 'test.txt')

    def load_file(file_path):
        images = []
        labels = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                img_path, label = line.strip().split('\t')
                full_path = os.path.join(dataset_path, img_path)

                # Try to load the image and skip if not found
                try:
                    image = Image.open(full_path).convert('L').resize((128, 32))  # Convert to grayscale
                    images.append(np.array(image))  # Convert image to NumPy array

                    # Convert label to a list of character indices
                    label_indices = [char_to_index.get(char, -1) for char in label]
                    labels.append(label_indices)  # Append the label as indices

                except FileNotFoundError:
                    print(f"Image not found: {full_path}, skipping.")
                    continue

        # Pad the labels to the max_length, use -1 for blank characters
        padded_labels = pad_sequences(labels, maxlen=max_length, padding='post', value=char_to_index[' '])  # Assuming space is the blank character

        return np.array(images), np.array(padded_labels)

    # Load train, validation, and test data
    train_images, train_labels = load_file(train_file)
    val_images, val_labels = load_file(val_file)
    test_images, test_labels = load_file(test_file)

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

# Load Characters Function
def load_chars(chars_file):
    with open(chars_file, 'r', encoding='utf-8') as f:
        chars = f.read().strip().splitlines()
    return np.array(chars)

# Set the path to your dataset
dataset_path = '/home/springschool_local/Shaukatali/USHWRdataset/UHWR'
chars_file = os.path.join(dataset_path, 'chars.txt')

# Load characters and create character index, adding a blank character
chars = load_chars(chars_file)
blank_char = ' '  # Define the blank character
chars = np.append(chars, blank_char)  # Add the blank character to the character array
char_to_index = {char: idx for idx, char in enumerate(chars)}
num_classes = len(chars)

print(f"Character to Index Mapping: {char_to_index}")
print(f"Number of Classes: {num_classes}")

# Calculate maximum label length from the dataset
def find_max_length(dataset_path):
    max_length = 0
    for file_name in ['train.txt', 'val.txt', 'test.txt']:
        file_path = os.path.join(dataset_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                _, label = line.strip().split('\t')
                max_length = max(max_length, len(label))
    return max_length

# Find the maximum label length in the dataset
max_length = find_max_length(dataset_path)
print(f"Maximum Label Length: {max_length}")

# Load the data with padded labels
(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_data(
    dataset_path, char_to_index, max_length
)

# One-hot encode the labels for categorical crossentropy
train_labels = to_categorical(train_labels, num_classes=num_classes)
val_labels = to_categorical(val_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

# Remove extra dimension in labels (squeeze to remove time step dimension)
train_labels = np.squeeze(train_labels, axis=1)
val_labels = np.squeeze(val_labels, axis=1)
test_labels = np.squeeze(test_labels, axis=1)

# Print shapes for verification
print(f"Train Labels Shape (After Squeezing): {train_labels.shape}")
print(f"Validation Labels Shape (After Squeezing): {val_labels.shape}")
print(f"Test Labels Shape (After Squeezing): {test_labels.shape}")

# Reshape images for LSTM (samples, time steps, height, width, channels)
X_train = train_images.reshape(train_images.shape[0], 1, 32, 128, 1)  # Change last dimension to 1 for grayscale
X_val = val_images.reshape(val_images.shape[0], 1, 32, 128, 1)
X_test = test_images.reshape(test_images.shape[0], 1, 32, 128, 1)

print(f"Train Images Shape (After Reshaping): {X_train.shape}")
print(f"Validation Images Shape (After Reshaping): {X_val.shape}")
print(f"Test Images Shape (After Reshaping): {X_test.shape}")

# Build the CNN-LSTM Model
model = Sequential()

# CNN Layers
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=(X_train.shape[1], 32, 128, 1)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(BatchNormalization()))

# Flatten and LSTM Layers
model.add(TimeDistributed(Flatten()))
model.add(LSTM(128, return_sequences=False))  # Change return_sequences to False

# Output layer (to match the number of classes)
model.add(Dense(num_classes, activation='softmax'))  # Softmax for one-hot encoded labels

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Summary of the model
model.summary()

# Set up model checkpoints and early stopping
checkpoint = ModelCheckpoint('cnn_lstm_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Train the model
history = model.fit(X_train, train_labels,
                    validation_data=(X_val, val_labels),
                    batch_size=32,
                    epochs=50,
                    callbacks=[checkpoint, early_stopping])

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(X_test, test_labels, verbose=1)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

