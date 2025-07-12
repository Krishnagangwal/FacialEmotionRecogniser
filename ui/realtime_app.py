# ui/realtime_app.py

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from keras.models import load_model

model = load_model("/Users/krishnagangwal/Documents/facial-emotion-analyzer/saved_model/fixed_model.keras", compile=False)
face_cascade = cv2.CascadeClassifier("main/haarcascade_frontalface_default.xml")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Title
st.set_page_config(page_title="Live Emotion Detector")
st.title("🎥 Real-Time Facial Emotion Detection")

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            preds = model.predict(roi, verbose=0)
            label = emotion_labels[np.argmax(preds)]

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return img

# Start webcam stream
webrtc_streamer(
    key="emotion",
    video_transformer_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)