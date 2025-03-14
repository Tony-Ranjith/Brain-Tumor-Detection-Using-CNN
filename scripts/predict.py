import cv2
import numpy as np
import tensorflow as tf

# Function to load and preprocess a single image
def load_and_preprocess_image(image_path, image_size=(128, 128)):
    # Load the image
    image = cv2.imread(image_path)
    
    # Resize image to match the model's input size
    image = cv2.resize(image, image_size)
    
    # Normalize the image (same preprocessing as during training)
    image = np.array(image, dtype="float32") / 255.0
    
    # Expand dimensions to match the input shape of the model (1, height, width, channels)
    image = np.expand_dims(image, axis=0)
    
    return image

# Function to predict if the image contains a tumor or not
def predict_image(image_path):
    # Load the trained model
    model = tf.keras.models.load_model('model/brain_tumor_model.h5')
    
    # Preprocess the image
    preprocessed_image = load_and_preprocess_image(image_path)
    
    # Make the prediction
    prediction = model.predict(preprocessed_image)
    
    # Convert prediction to label (0 or 1)
    predicted_label = 1 if prediction > 0.5 else 0
    
    # Interpret the result
    if predicted_label == 1:
        prediction_text = "Tumor detected"
        color = (0, 0, 255)  # Red for tumor detected
    else:
        prediction_text = "No tumor detected"
        color = (0, 255, 0)  # Green for no tumor

    # Load the original image (to show it with the prediction)
    original_image = cv2.imread(image_path)
    
    # Put the prediction text on the image
    cv2.putText(original_image, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Show the image with the prediction
    cv2.imshow("Predicted Result", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Also print the prediction in the terminal
    print(f"Prediction: {prediction_text}")

# Run the prediction on a new image
if __name__ == "__main__":
    # Path to the new image
    image_path = 'images_for_prediction/6 no.jpg'  # Update with the actual path to the new image
    
    # Predict
    predict_image(image_path)

