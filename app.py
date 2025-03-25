import streamlit as st
import numpy as np
import cv2
from keras.models import load_model

# Load the trained model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Save the model as 'mymodel.h5'
model.save('mymodel.h5')

# Load the trained model
model = load_model('mymodel.h5')


# Function to predict fault
def predict_fault(image):
    # Preprocess the image as required by the model
    image = cv2.resize(image, (300, 300))  # Resize to match model input
    image = image / 255.0  # Rescale pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    prediction = model.predict(image)
    return prediction

# Streamlit app layout
st.markdown("""
    <style>
    body {
        background-image: url('https://images.pexels.com/photos/3690392/pexels-photo-3690392.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2');
        background-size: cover;
        color: white;
    }
    </style>
    <h1 style='text-align: center;'>Rail Track Fault Detection System</h1>
""", unsafe_allow_html=True)

st.write("<h2 style='text-align: center;'>Upload a track image to check for faults.</h2>", unsafe_allow_html=True)

st.sidebar.header("Options")
st.sidebar.write("Use this application to detect faults in railway tracks.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed", help="Upload an image of the railway track.")

if uploaded_file is not None:    
    # Read the image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption='Uploaded Image', use_column_width=True)  # Display the uploaded image

st.markdown("<hr>", unsafe_allow_html=True)  # Add a horizontal line for separation

if uploaded_file is not None:
    with st.spinner("Processing..."):  # Show a loading spinner
        # Predict fault
        prediction = predict_fault(image)
        
        # Display result
        if prediction[0][0] > 0.5:  # Assuming binary classification
            st.success("Fault detected! 🚨")  # Enhanced message with emoji
        else:
            st.success("No fault detected. ✅")  # Enhanced message with emoji
