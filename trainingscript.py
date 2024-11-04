import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, BatchNormalization, Flatten, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

# Load Data Function
def load_data(dataset_path):
    train_file = os.path.join(dataset_path, 'train.txt')
    val_file = os.path.join(dataset_path, 'val.txt')
    test_file = os.path.join(dataset_path, 'test.txt')

    def load_file(file_path):
        images = []
        labels = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                img_path, label = line.strip().split('\t')
                full_path = os.path.join(dataset_path, 'images', img_path)
                
                # Try to load the image and skip if not found
                try:
                    image = Image.open(full_path).convert('RGB').resize((128, 32))  # Resize image
                    images.append(np.array(image))  # Convert image to NumPy array
                    labels.append(label)              # Append label
                except FileNotFoundError:
                    print(f"Image not found: {full_path}, skipping.")
                    continue

        return np.array(images), np.array(labels)


    # Load train, validation, and test data
    train_images, train_labels = load_file(train_file)
    val_images, val_labels = load_file(val_file)
    test_images, test_labels = load_file(test_file)

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

# Load Characters Function
def load_chars(chars_file):
    with open(chars_file, 'r', encoding='utf-8') as f:
        chars = f.read().strip().splitlines()  # Read and split into lines
    return np.array(chars)

# Pad Labels Function
def pad_labels(labels, char_to_index, max_length):
    y = [[char_to_index[char] for char in label] for label in labels]
    y_padded = pad_sequences(y, maxlen=max_length, padding='post', value=-1)  # Use -1 for padding
    return y_padded

# Set the path to your dataset
dataset_path = '/home/springschool_local/Shaukatali/USHWRdataset/UHWR'
chars_file = os.path.join(dataset_path, 'chars.txt')

# Load the data
(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_data(dataset_path)

# Load characters and create character index
chars = load_chars(chars_file)
char_to_index = {char: idx for idx, char in enumerate(chars)}
num_classes = len(chars)
print(char_to_index)
print(num_classes)
# Set maximum length for padding
# Ensure that train_labels, val_labels, and test_labels are not empty before calculating max_length
if len(train_labels) > 0 and len(val_labels) > 0 and len(test_labels) > 0:
    max_length = max(
        max(len(label) for label in train_labels), 
        max(len(label) for label in val_labels), 
        max(len(label) for label in test_labels)
    )
else:
    raise ValueError("One or more label sequences are empty. Please check your dataset.")

# Pad labels
y_train_padded = pad_labels(train_labels, char_to_index, max_length)
y_val_padded = pad_labels(val_labels, char_to_index, max_length)
y_test_padded = pad_labels(test_labels, char_to_index, max_length)

# Reshape images for LSTM (number of samples, time steps, height, width, channels)
# Assuming you want to treat each image as one time step
X_train = train_images.reshape(train_images.shape[0], 1, 128, 128, 3)
X_val = val_images.reshape(val_images.shape[0], 1, 128, 128, 3)
X_test = test_images.reshape(test_images.shape[0], 1, 128, 128, 3)

# Model Definition
model = Sequential()

# CNN Layers
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])))
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
model.add(LSTM(128, return_sequences=False))
model.add(Dense(num_classes, activation='sigmoid'))  # Use sigmoid for multi-label classification

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
model.fit(X_train, y_train_padded, validation_data=(X_val, y_val_padded), epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test_padded)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

