import os
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, Conv2D
from tensorflow.keras.optimizers import Adam

# Step 1: Load CSV data
csv_file = 'delhi_data_2019.csv'
df = pd.read_csv(csv_file)
print(df.head())

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

# Convert 'year', 'month', 'day' columns to datetime
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

# Apply load_sentinel_image function to create 'image' column
df['image'] = df['date'].apply(lambda x: load_sentinel_image(x))

# Filter out rows with missing images
df = df[df['image'].notnull()]

# Step 3: Feature Engineering
def extract_image_features(image):
    # Example: Resize image to (64, 64)
    image_resized = tf.image.resize(image, (64, 64))
    return image_resized

# Apply feature extraction to each image and create a new column
df['image_features'] = df['image'].apply(lambda x: extract_image_features(x))

# Step 4: Prepare sequences of images
sequence_length = 10  # Length of the sequence for LSTM

# Create sequences of images
sequences = []
targets = []

for i in range(len(df) - sequence_length):
    seq = np.stack(df['image_features'].iloc[i:i+sequence_length].values)
    target = df['image_features'].iloc[i + sequence_length]
    sequences.append(seq)
    targets.append(target)

X = np.array(sequences)
y = np.array(targets)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Define the ConvLSTM model
def build_convlstm_model(input_shape):
    input_seq = Input(shape=input_shape)

    # ConvLSTM2D layers
    x = ConvLSTM2D(32, (3, 3), activation='relu', padding='same', return_sequences=True)(input_seq)
    x = ConvLSTM2D(64, (3, 3), activation='relu', padding='same', return_sequences=False)(x)
    
    # Conv2D layer for output
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(input_seq, x)
    return model

input_shape = (sequence_length, 64, 64, 1)  # Example input shape
model = build_convlstm_model(input_shape)
model.compile(optimizer=Adam(), loss='mse', metrics=['mse'])

# Step 6: Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Save the trained model
model.save('trained_convlstm_model.h5')

# Step 7: Evaluate the model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate R2 score
train_r2 = r2_score(y_train.flatten(), y_pred_train.flatten())
test_r2 = r2_score(y_test.flatten(), y_pred_test.flatten())
print(f"Training R2 Score: {train_r2}")
print(f"Testing R2 Score: {test_r2}")
