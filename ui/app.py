import streamlit as st
import cv2
from PIL import Image
import numpy as np
from src.detect_emotion import predict_emotion

st.set_page_config(page_title="Facial Emotion Detector", layout="wide")

# Dark mode UI (fixed background + white text)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1d3a; /* Dark blue */
        color: white;
    }

    /* General text and input styling */
    label, .stTextInput > div > div > input, .stFileUploader label div {
        color: white !important;
    }

    /* Mood Suggestion box style (overriding st.info) */
    .stAlert {
        background-color: #cce5ff !important;  /* Light blue */
        color: black !important;
        border: 1px solid #b8daff;
        border-radius: 8px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True

)
st.title(" Real-Time Facial Emotion Recognition")
st.write("Upload a photo and get emotion predictions for detected faces.")

# 👇 Only keep image upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = predict_emotion(image)

    # Draw boxes and show results
    for res in results:
        x, y, w, h = res['box']
        label = f"{res['emotion']} ({res['confidence']*100:.0f}%)"
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    st.image(image, caption="Predicted Image", use_column_width=True)

    if results:
        mood = results[0]['emotion']
        st.subheader(f"Mood-based Suggestion ")
        suggestions = {
            "Happy": "Keep smiling! Here's a great playlist: [Spotify](https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC)",
            "Sad": "Take a walk, talk to someone. You're not alone 💙",
            "Angry": "Take deep breaths. Here's a calming video: [YouTube](https://youtu.be/1vx8iUvfyCY)",
            "Neutral": "Balanced mind, balanced life. Stay aware.",
            "Surprise": "Whoa! Did something unexpected happen? 😲",
            "Fear": "You’re safe. Let someone know how you feel.",
            "Disgust": "Clean environment helps. Maybe take a breather."
        }
        st.info(suggestions.get(mood, "Stay mindful."))