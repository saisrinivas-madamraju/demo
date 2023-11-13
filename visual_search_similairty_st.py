import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


st.title('Similarity Search')

# Load the model
model = tf.keras.models.load_model('vs_similarity.h5')

# Allow image upload
uploaded_file = st.file_uploader("choose an image which you want to clasify", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load image and convert to numpy array
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))
    image = np.array(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    # Make prediction
    predictions = model.predict(image)
    # Get index of top prediction
    predicted_class = np.argmax(predictions[0])

  
  # Map index to class name Dress, Hat, Long Sleeve,  Shoes, Tshirt
  
    class_names= ['Dress' ,'Hat' ,'Longsleeve' ,'Shoes' ,'T-Shirt']

    predicted_class_name = class_names[predicted_class]
    
    # Show image and prediction
    st.image(image, use_column_width=True)
    st.write(f'it is a type of: {predicted_class_name}')
    # Make prediction 

    # Get prediction probabilities for each class
    prediction_probs = predictions[0]

    # Display probabilities for each class  
    for i in range(len(class_names)):
      st.write(f'{class_names[i]}: {100 * prediction_probs[i]:.2f}%') 
    # Or to return as dictionary:
    pred_dict = dict(zip(class_names, prediction_probs))