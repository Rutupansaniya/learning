import numpy as np
import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
from dotenv import load_dotenv 
import os


def load_image(img):

    #img = Image.open(image_file)
    
    
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    #load model keras_model.h5
    model = load_model('keras_model.h5')

    #load lables names
    class_names = open('labels.txt','r').readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = img.convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    #make prediction
    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    #print("Class:", class_name[2:], end="")
    #print("Confidence Score:", confidence_score)

    return class_name, confidence_score

st.set_page_config(layout='wide')

st.title('leather')

st.write('This is a leather classification model')

input_img = st.file_uploader("Enter your image", type=['jpg', 'png', 'jpeg'])

if input_img is not None:
    if st.button("Classify"):
        
        col1, col2 = st.columns([1,1])
        with col1:
            st.info('upload image')
            st.image(input_img, use_container_width=True)

        with col2:
            st.info("Your Result")
            image_file = Image.open(input_img)
            label, confidence = load_image(image_file)
            st.write(label)
            st.write(confidence)
            
