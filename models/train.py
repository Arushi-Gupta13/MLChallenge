import numpy as np
import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

def load_data_from_directory(directory):
    x_data = []
    y_data = []

    for file_name in sorted(os.listdir(directory)):
        if file_name.endswith('.npy') and not file_name.endswith('_labels.npy'):
            image_file = os.path.join(directory, file_name)
            label_file = image_file.replace('.npy', '_labels.npy')
            
            if os.path.exists(label_file):
                images = np.load(image_file)
                labels = np.load(label_file)
                
                x_data.append(images)
                y_data.append(labels)
            else:
                print(f"Label file {os.path.basename(label_file)} not found, skipping.")
    
    if x_data and y_data:
        return np.concatenate(x_data), np.concatenate(y_data)
    else:
        raise ValueError("No label files found in the specified directory.")

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main(npy_folder):
    print("Starting data load...")
    try:
        x_data, y_data = load_data_from_directory(npy_folder)
        print(f"Data shape: {x_data.shape}")
    except ValueError as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    num_classes = y_data.shape[1]  # Assuming y_data is one-hot encoded
    input_shape = x_data.shape[1:]  # Shape of images (height, width, channels)

    model = create_model(input_shape, num_classes)
    
    # Split data into training and validation sets
    split_index = int(0.9 * len(x_data))
    x_train, x_val = x_data[:split_index], x_data[split_index:]
    y_train, y_val = y_data[:split_index], y_data[split_index:]

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

    # Save the model
    model.save('trained_model.h5')
    print("Model training complete and saved to 'trained_model.h5'")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <npy_folder>")
        sys.exit(1)
    
    npy_folder = sys.argv[1]
    main(npy_folder)
