import re
import constants
import os
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path
from functools import partial
import urllib
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

def common_mistake(unit):
    if unit in constants.allowed_units:
        return unit
    if unit.replace('ter', 'tre') in constants.allowed_units:
        return unit.replace('ter', 'tre')
    if unit.replace('feet', 'foot') in constants.allowed_units:
        return unit.replace('feet', 'foot')
    return unit

def parse_string(s):
    s_stripped = "" if s == None or str(s) == 'nan' else s.strip()
    if s_stripped == "":
        return None, None
    pattern = re.compile(r'^-?\d+(\.\d+)?\s+[a-zA-Z\s]+$')
    if not pattern.match(s_stripped):
        raise ValueError("Invalid format in {}".format(s))
    parts = s_stripped.split(maxsplit=1)
    number = float(parts[0])
    unit = common_mistake(parts[1])
    if unit not in constants.allowed_units:
        raise ValueError("Invalid unit [{}] found in {}. Allowed units: {}".format(
            unit, s, constants.allowed_units))
    return number, unit

def create_placeholder_image(image_save_path):
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
    except Exception as e:
        print(f"Error creating placeholder image: {str(e)}")

def download_image(image_link, save_folder, retries=3, delay=3):
    if not isinstance(image_link, str):
        print("Invalid image link type:", image_link)
        return

    filename = Path(image_link).name
    image_save_path = os.path.join(save_folder, filename)

    if os.path.exists(image_save_path):
        print(f"Image {filename} already exists, skipping download.")
        return

    for _ in range(retries):
        try:
            print(f"Attempting to download {image_link}")
            urllib.request.urlretrieve(image_link, image_save_path)
            print(f"Downloaded {image_link} successfully")
            return
        except Exception as e:
            print(f"Failed to download {image_link}, retrying... Error: {str(e)}")
            time.sleep(delay)
    
    print(f"Failed to download {image_link} after {retries} attempts. Creating placeholder.")
    create_placeholder_image(image_save_path)  # Create a black placeholder image for invalid links/images

def download_images_in_batches(image_links, download_folder, batch_size=100, max_workers=10):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    total_links = len(image_links)
    # Limit the number of links to the first 1000
    image_links = image_links[:1000]
    total_links = len(image_links)
    
    for i in range(0, total_links, batch_size):
        batch_links = image_links[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} of {total_links // batch_size + 1}")

        # ThreadPoolExecutor for multithreading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Partial function to download images with fixed folder and retry logic
            download_image_partial = partial(download_image, save_folder=download_folder, retries=3, delay=3)
            
            # Submit download tasks to the thread pool
            futures = {executor.submit(download_image_partial, link): link for link in batch_links}
            
            # Display progress using tqdm
            for future in tqdm(as_completed(futures), total=len(batch_links)):
                try:
                    future.result()  # We can handle exceptions here if needed
                except Exception as e:
                    print(f"Error downloading image: {futures[future]} -> {e}")

        # Optionally, you can add a delay between batches to avoid overloading the server
        time.sleep(5)

if __name__ == "__main__":
    # Example usage
    train_csv = "dataset/train.csv"  # Check if this path is correct
    image_directory = "data/images"  # Ensure this directory exists or will be created

    # Load image URLs from the CSV file
    try:
        df = pd.read_csv(train_csv)
        image_links = df['image_link'].tolist()  # Check if the column exists and is loaded properly
        print(f"Found {len(image_links)} image links.")
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        exit(1)

    # Download images with multithreading in batches
    download_images_in_batches(image_links, image_directory, batch_size=100, max_workers=10)
