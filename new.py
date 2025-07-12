from keras.models import load_model
import tensorflow as tf

# Disable Keras warnings for optimizer mismatch
model = load_model("saved_model/best_emotion_model.keras", compile=False)

# Save in a new format that fixes deserialization issues
model.save("saved_model/fixed_model.keras", save_format="keras")



