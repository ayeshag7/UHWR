import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

        # Pad the labels to the max_length
        padded_labels = pad_sequences(labels, maxlen=max_length, padding='post', value=-1)

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

# Load characters and create character index
chars = load_chars(chars_file)
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

# Print shapes for verification
print(f"Train Images Shape: {train_images.shape}, Train Labels Shape: {train_labels.shape}")
print(f"Validation Images Shape: {val_images.shape}, Validation Labels Shape: {val_labels.shape}")
print(f"Test Images Shape: {test_images.shape}, Test Labels Shape: {test_labels.shape}")

# Reshape images for LSTM (samples, time steps, height, width, channels)
X_train = train_images.reshape(train_images.shape[0], 1, 32, 128, 1)  # Change last dimension to 1 for grayscale
X_val = val_images.reshape(val_images.shape[0], 1, 32, 128, 1)
X_test = test_images.reshape(test_images.shape[0], 1, 32, 128, 1)

print(f"Train Images Shape (After Reshaping): {X_train.shape}")
print(f"Validation Images Shape (After Reshaping): {X_val.shape}")
print(f"Test Images Shape (After Reshaping): {X_test.shape}")

