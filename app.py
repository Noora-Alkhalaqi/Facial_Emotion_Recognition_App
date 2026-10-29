import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from keras.models import load_model

class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

@st.cache_resource
def get_model():
    return load_model("emotion_model.keras")

model = get_model()

def predict(img):
    gray = img.convert("L").resize((48, 48))      
    arr = np.array(gray) / 255                    
    arr = arr.reshape(1, 48, 48)                  
    probs = model.predict(arr, verbose=0)[0]
    return probs

def show_result(img):
    probs = predict(img)
    st.image(img.convert("L").resize((192, 192)), caption="What the model sees (48x48 grayscale)")
    st.subheader(f"Prediction: {class_names[probs.argmax()]}")
    st.write(f"Confidence: {probs.max():.1%}")
    st.bar_chart(pd.Series(probs, index=class_names))

st.title("Facial Emotion Recognition")
st.write("Upload a face image or take one with your webcam.")

tab1, tab2 = st.tabs(["Upload image", "Webcam"])

with tab1:
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if file is not None:
        show_result(Image.open(file))

with tab2:
    shot = st.camera_input("Take a picture")
    if shot is not None:
        show_result(Image.open(shot))