import os
import cv2
import numpy as np
import tensorflow as tf

# Function to load and preprocess test images (same as in train_model)
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

# Function to test the model
def test_model():
    # Set data directories for test data
    tumor_dir = 'data/tumor/'  # Make sure the test set is organized
    no_tumor_dir = 'data/no_tumor/'

    # Load test images and their corresponding labels
    images, labels = load_images(tumor_dir, no_tumor_dir)

    # Load the saved model
    model = tf.keras.models.load_model('model/brain_tumor_model.h5')

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(images, labels)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Predict on the test set (optional: to see individual predictions)
    predictions = model.predict(images)
    predictions = [1 if p > 0.5 else 0 for p in predictions]  # Convert probabilities to labels (0 or 1)

    # Print the comparison of predicted labels and actual labels
    for i in range(len(labels)):
        print(f"True label: {labels[i]}, Predicted label: {predictions[i]}")

if __name__ == "__main__":
    test_model()

