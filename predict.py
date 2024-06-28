import os
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('trained_convlstm_model.h5')

# Step 2: Load Sentinel images
image_folder = r"ImageData\2019\Delhi_NO2"

def load_sentinel_image(date):
    image_name = f'DelhiNO2{date.strftime("%Y-%m-%d")}.tif'
    image_path = os.path.join(image_folder, image_name)
    try:
        image = imread(image_path)
        # Convert to grayscale
        image = rgb2gray(image)
        # Normalize the image
        image = image / 255.0
        # Expand dimensions to add a channel
        image = np.expand_dims(image, axis=-1)
        return image
    except FileNotFoundError:
        print(f"Warning: File not found - {image_path}")
        return None

def extract_image_features(image):
    # Example: Resize image to (64, 64)
    image_resized = tf.image.resize(image, (64, 64))
    return image_resized

# Step 8: Predict NO2 image for a specific date
def predict_no2_image(date_of_interest, sequence_length):
    # Generate sequence dates
    sequence_dates = [date_of_interest - pd.Timedelta(days=i) for i in range(sequence_length)]
    sequence_dates.reverse()
    
    # Load sequence images
    sequence_images = [load_sentinel_image(date) for date in sequence_dates]
    
    if None not in sequence_images:
        sequence_images = [extract_image_features(img) for img in sequence_images]
        sequence_images = np.array(sequence_images).reshape(1, sequence_length, 64, 64, 1)  # Reshape for prediction
        predicted_image = model.predict(sequence_images)
        predicted_image = predicted_image.reshape(64, 64)  # Reshape to original image shape
        
        return predicted_image
    else:
        print(f"Not all images found for the sequence. Prediction cannot be made.")
        return None

date_of_interest = pd.to_datetime('2019-01-01')
predicted_image = predict_no2_image(date_of_interest, sequence_length=10)

if predicted_image is not None:
    print(f"Predicted NO2 image for {date_of_interest.date()}:")
    plt.imshow(predicted_image, cmap='gray')
    plt.title(f'Predicted NO2 Image for {date_of_interest.date()}')
    plt.colorbar()
    plt.savefig(f'predicted_NO2_image_{date_of_interest.date()}.png')
    plt.show()
else:
    print(f"No image found for {date_of_interest.date()}. Prediction cannot be made.")
