import os
import cv2
import numpy as np
from keras.models import load_model

# Get root directory dynamically
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Correct absolute paths
model_path = os.path.join(ROOT_DIR, "saved_model", "best_emotion_model.keras")
cascade_path = os.path.join(ROOT_DIR, "main", "haarcascade_frontalface_default.xml")

# Load model and face detector
model = load_model(model_path)
face_cascade = cv2.CascadeClassifier(cascade_path)

# Check if Haar cascade loaded correctly
if face_cascade.empty():
    raise IOError("❌ Failed to load Haar Cascade. Check the file path.")

model = load_model("saved_model/best_emotion_model.keras", compile=False)
face_cascade = cv2.CascadeClassifier("main/haarcascade_frontalface_default.xml")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def predict_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    results = []

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        preds = model.predict(roi, verbose=0)
        emotion = emotion_labels[np.argmax(preds)]
        confidence = np.max(preds)

        results.append({
            'emotion': emotion,
            'confidence': round(float(confidence), 2),
            'box': (x, y, w, h)
        })

    return results