import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def preprocess_image(image):
    # Resize and convert to grayscale
    image = image.resize((28, 28)).convert('L')

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Use StandardScaler to scale the pixel values
    scale = StandardScaler()
    image_array_scaled = scale.fit_transform(image_array.astype(float))

    # Reshape for model input
    image_array_scaled = image_array_scaled.reshape(-1, 28, 28, 1)
    
    return image_array_scaled

def predict_digit(image):
    img_array = preprocess_image(image)
    
    loaded_model = tf.keras.models.load_model('Saved Model')

    # Make predictions on the new data
    predictions = loaded_model.predict(img_array)

    # Get the predicted class label (assuming it's a classification task)
    predicted_class = np.argmax(predictions)

    return predicted_class  # Return the predicted class label instead of printing it

def main():
    st.title('Hand Digit Recognizer')

    # Upload image
    uploaded_image = st.file_uploader("Upload a hand digit image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        if st.button('Predict'):
            digit = predict_digit(image)
            st.success(f'Predicted digit: {digit}')

if __name__ == '__main__':
    main()