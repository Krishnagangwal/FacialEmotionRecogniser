from keras.models import load_model

# Load the original model (you already have this one)
model = load_model("saved_model/best_emotion_model.keras", compile=False)

# Save a cleaned-up version
model.save("saved_model/fixed_model.keras", save_format="keras")

print("✅ Fixed model saved as fixed_model.keras")