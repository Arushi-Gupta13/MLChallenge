import numpy as np
import os
import argparse
import tensorflow as tf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_from_directory(directory):
    x_data = []
    y_data = []
    files_processed = 0
    labels_found = 0

    logging.info(f"Attempting to load data from directory: {directory}")

    # Check if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The specified directory does not exist: {directory}")

    # Iterate through all files in the directory
    for file_name in sorted(os.listdir(directory)):
        if file_name.endswith('.npy') and not file_name.endswith('_labels.npy'):
            image_file = os.path.join(directory, file_name)
            
            # Attempt to load the image data
            try:
                images = np.load(image_file)
                x_data.append(images)
                files_processed += 1
                logging.info(f"Loaded image data from: {file_name}")
                
                # Check if a corresponding label file exists
                label_file = image_file.replace('.npy', '_labels.npy')
                if os.path.exists(label_file):
                    labels = np.load(label_file)
                    y_data.append(labels)
                    labels_found += 1
                    logging.info(f"Loaded labels from: {os.path.basename(label_file)}")
                else:
                    # Handle missing label files
                    logging.warning(f"Label file {os.path.basename(label_file)} not found, generating synthetic labels.")
                    # Generate synthetic labels (e.g., zeros or other default values)
                    # Adjust the synthetic label shape based on your model's requirement
                    synthetic_labels = np.zeros((images.shape[0], 1))  # Adjust shape if needed
                    y_data.append(synthetic_labels)
                
            except Exception as e:
                logging.error(f"Error loading data from {file_name}: {e}")

    # Convert lists to numpy arrays
    if x_data:
        x_data = np.concatenate(x_data)
    if y_data:
        y_data = np.concatenate(y_data)

    logging.info(f"Processed {files_processed} image files.")
    logging.info(f"Found labels for {labels_found} image files.")
    logging.info(f"Generated synthetic labels for {files_processed - labels_found} image files.")

    if x_data.size == 0:
        raise ValueError("No image data found in the specified directory.")
    
    return x_data, y_data

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, x_data, y_data, epochs=5, batch_size=32):
    try:
        history = model.fit(x_data, y_data, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        return history
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

def save_model(model, filename='trained_model.h5'):
    try:
        model.save(filename)
        logging.info(f"Model saved as '{filename}'")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def main(data_directory):
    logging.info("Starting data load...")
    try:
        x_data, y_data = load_data_from_directory(data_directory)
        
        logging.info(f"Data shape: {x_data.shape}")
        logging.info(f"Labels shape: {y_data.shape}")

        model = build_model(input_shape=x_data.shape[1:])
        logging.info("Model built successfully.")

        logging.info("Starting model training...")
        history = train_model(model, x_data, y_data)
        
        logging.info("Training completed. Saving model...")
        save_model(model)

        # Optional: You can add code here to plot training history or perform model evaluation
        
    except Exception as e:
        logging.error(f"An error occurred during the process: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with image data.')
    parser.add_argument('data_directory', type=str, help='Directory containing the preprocessed image data.')
    args = parser.parse_args()
    
    main(args.data_directory)