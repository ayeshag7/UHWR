import os
import numpy as np
from PIL import Image

# Function to load images
def load_images(dataset_path, file_list):
    images = []
    for img_path in file_list:
        full_path = os.path.join(dataset_path, img_path)

        try:
            image = Image.open(full_path).convert('L').resize((128, 32))  # Grayscale and resize
            images.append(np.array(image) / 255.0)  # Normalize
        except FileNotFoundError:
            print(f"Image not found: {full_path}, skipping.")
            continue

    images = np.array(images)
    print(f"Loaded {len(images)} images with shape: {images.shape}")
    return images

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
def load_file(file_path, dataset_path):
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

# Load Data Function (separate loading of images and labels)
def load_data(dataset_path, chars):
    train_file = os.path.join(dataset_path, 'train.txt')
    val_file = os.path.join(dataset_path, 'val.txt')
    test_file = os.path.join(dataset_path, 'test.txt')

    # Load file paths and labels
    train_images_paths, train_labels = load_file(train_file, dataset_path)
    val_images_paths, val_labels = load_file(val_file, dataset_path)
    test_images_paths, test_labels = load_file(test_file, dataset_path)

    # Find maximum label length across all datasets (train, val, test)
    max_label_len = max(
        find_max_label_length(train_labels),
        find_max_label_length(val_labels),
        find_max_label_length(test_labels)
    )

    # Load and one-hot encode labels to ensure uniform length and encoding
    train_labels_encoded = load_labels(train_labels, chars, max_label_len)
    val_labels_encoded = load_labels(val_labels, chars, max_label_len)
    test_labels_encoded = load_labels(test_labels, chars, max_label_len)

    # Load actual images using paths
    train_images = load_images(dataset_path, train_images_paths)
    val_images = load_images(dataset_path, val_images_paths)
    test_images = load_images(dataset_path, test_images_paths)

    return (train_images, train_labels_encoded), (val_images, val_labels_encoded), (test_images, test_labels_encoded)

# Load Characters Function
def load_chars(chars_file):
    with open(chars_file, 'r', encoding='utf-8') as f:
        chars = f.read().strip().splitlines()
    chars = [''] + chars  # Add blank character at index 0
    if ' ' not in chars:
        chars.append(' ')  # Add space character if missing
    return np.array(chars)

# Main function to call all required functions
def main():
    # Specify paths
    dataset_path = '/home/springschool_local/Shaukatali/USHWRdataset/UHWR'  # Replace with actual path to the dataset
    chars_file = '/home/springschool_local/Shaukatali/USHWRdataset/UHWR/chars.txt'      # Replace with actual path to the char<<<<acter file

    # Load the characters
    chars = load_chars(chars_file)
    print(f"Loaded characters with length: {len(chars)}")

    # Load the dataset (images and one-hot encoded labels)
    (train_images, train_labels_encoded), (val_images, val_labels_encoded), (test_images, test_labels_encoded) = load_data(dataset_path, chars)

    # Print the shapes of the encoded labels
    print(f"Train labels encoded shape: {train_labels_encoded.shape}")
    print(f"Validation labels encoded shape: {val_labels_encoded.shape}")
    print(f"Test labels encoded shape: {test_labels_encoded.shape}")
if __name__ == '__main__':
    main()

