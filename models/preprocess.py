import os
from PIL import Image
import numpy as np
import pandas as pd
import sys

def preprocess_image(image_path, target_size=(224, 224)):
    try:
        with Image.open(image_path) as img:
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
            if img_array.shape[-1] != 3:  # Ensure image has 3 channels (RGB)
                raise ValueError(f"Image {image_path} does not have 3 channels")
            return img_array
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def preprocess_batch(image_paths, target_size=(224, 224)):
    images = []
    for image_path in image_paths:
        img_array = preprocess_image(image_path, target_size)
        if img_array is not None:
            images.append(img_array)
        else:
            print(f"Skipping {image_path} due to preprocessing error.")
    
    if not images:
        raise ValueError("No valid images found for preprocessing.")
    
    return np.array(images)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <csv_file> <output_folder>")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_folder = sys.argv[2]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    try:
        df = pd.read_csv(csv_file)
        image_links = df['image_link'].tolist()
        image_paths = [os.path.join('data/images', os.path.basename(link)) for link in image_links]
        
        if not image_paths:
            print(f"No images found in the CSV file.")
            sys.exit(1)
        
        # Preprocess images in batches and save them
        batch_size = 100  # Adjust batch size as needed
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            preprocessed_images = preprocess_batch(batch_paths)
            
            # Save preprocessed images as npy files
            output_file = os.path.join(output_folder, f'batch_{i // batch_size}.npy')
            np.save(output_file, preprocessed_images)
            print(f"Saved batch {i // batch_size} of images to {output_file}.")
    
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        sys.exit(1)
