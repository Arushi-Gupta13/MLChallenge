import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

def load_data(npy_folder):
    data_files = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]
    
    if not data_files:
        raise ValueError("No .npy files found in the specified directory.")
    
    data = []
    
    for file_name in data_files:
        file_path = os.path.join(npy_folder, file_name)
        print(f"Loading data from: {file_path}")
        data.append(np.load(file_path))
    
    # Concatenate all data files
    data = np.concatenate(data, axis=0)
    
    return data

def build_model(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # Update the number of classes as needed
    ])
    
    model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy',  # Update if using one-hot encoding
                  metrics=['accuracy'])
    return model

def main():
    npy_folder = 'output/preprocessed_images'
    
    try:
        data = load_data(npy_folder)
        print(f"Data shape: {data.shape}")
    except ValueError as e:
        print(f"Error loading data: {e}")
        return

    input_shape = data.shape[1:]  # Adjust if necessary
    model = build_model(input_shape)
    
    # For demonstration: print model summary
    model.summary()

if __name__ == '__main__':
    main()
