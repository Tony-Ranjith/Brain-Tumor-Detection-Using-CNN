import os
import cv2
import numpy as np
import tensorflow as tf

# Function to load and preprocess images
def load_images(tumor_dir, no_tumor_dir, image_size=(128, 128)):
    images = []
    labels = []

    # Load tumor images
    for filename in os.listdir(tumor_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(tumor_dir, filename)
            image = cv2.imread(img_path)
            image = cv2.resize(image, image_size)  # Resize image
            images.append(image)
            labels.append(1)  # Label for tumor

    # Load no tumor images
    for filename in os.listdir(no_tumor_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(no_tumor_dir, filename)
            image = cv2.imread(img_path)
            image = cv2.resize(image, image_size)  # Resize image
            images.append(image)
            labels.append(0)  # Label for no tumor

    # Convert images to numpy array and normalize to [0, 1]
    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels, dtype="float32")

    return images, labels

# Function to build the neural network model
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main function to train the model
def train_model():
    # Set data directories
    tumor_dir = 'data/tumor/'
    no_tumor_dir = 'data/no_tumor/'
    
    # Load and preprocess images
    images, labels = load_images(tumor_dir, no_tumor_dir)
    
    # Split the dataset into training and validation sets
    validation_split = 0.2
    split_idx = int(len(images) * (1 - validation_split))
    
    X_train, X_val = images[:split_idx], images[split_idx:]
    y_train, y_val = labels[:split_idx], labels[split_idx:]
    
    # Build the model
    input_shape = X_train.shape[1:]  # shape (height, width, channels)
    model = build_model(input_shape)

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

    # Save the model
    model.save('model/brain_tumor_model.h5')
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()



