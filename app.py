import os
import json

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np

app = Flask(__name__)

# Reject uploads larger than 8 MB
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# Resolve paths relative to this file so the app works from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_KERAS = os.path.join(BASE_DIR, "model", "plant_model.keras")
MODEL_PATH_H5 = os.path.join(BASE_DIR, "model", "plant_model.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model", "class_names.json")

# Load model – prefer .keras, fall back to .h5, then fall back to temp model if OneDrive is broken
try:
    if os.path.exists(MODEL_PATH_KERAS):
        try:
            model = load_model(MODEL_PATH_KERAS)
        except (ValueError, OSError):
            model = load_model(MODEL_PATH_H5)
    else:
        model = load_model(MODEL_PATH_H5)
except Exception as e:
    print(f"Warning: Could not load real model due to: {e}")
    print("Falling back to temporary mock model...")
    temp_model_path = os.path.join(BASE_DIR, "model", "temp_plant_model.keras")
    model = load_model(temp_model_path)

# Load class names (index order must match the model's output layer)
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# Suggestions for each class

suggestions = {

    "Potato_Early_Blight": 
    "Early blight is caused by fungal infection. Remove and destroy infected leaves immediately to prevent spread. Apply recommended fungicides such as chlorothalonil or mancozeb. Ensure proper crop rotation and avoid overhead irrigation to reduce moisture on leaves.",

    "Potato_Healthy": 
    "The plant is healthy. Maintain proper agricultural practices such as balanced fertilization, regular watering, and adequate sunlight. Monitor periodically for early signs of disease or pest infestation.",

    "Potato_Late_Blight": 
    "Late blight is a serious fungal disease. Remove and destroy infected plants immediately. Avoid excess moisture and improve field drainage. Apply fungicides like metalaxyl or copper-based sprays and maintain proper spacing for airflow.",

    "Tomato___Bacterial_spot": 
    "Bacterial spot spreads through water and contaminated tools. Use disease-free seeds and resistant varieties. Avoid overhead watering and practice crop rotation. Apply copper-based bactericides and maintain proper field hygiene.",

    "Tomato___Early_blight": 
    "Early blight affects older leaves first. Remove infected leaves and apply fungicides such as mancozeb. Ensure proper spacing and air circulation. Mulching can help prevent soil-borne spores from reaching leaves.",

    "Tomato___healthy":
    "The plant is healthy. Ensure consistent watering, sufficient sunlight, and proper nutrient supply. Regularly inspect plants to detect any early signs of disease or pest activity.",

    "Tomato___Late_blight": 
    "Late blight spreads rapidly in cool, humid conditions. Remove infected plants immediately and destroy them. Apply copper-based fungicides and avoid watering leaves. Improve air circulation and avoid overcrowding.",

    "Tomato___Leaf_Mold": 
    "Leaf mold develops in high humidity conditions. Improve ventilation and reduce humidity levels. Avoid wetting leaves during irrigation. Apply appropriate fungicides and remove infected foliage.",

    "Tomato___Septoria_leaf_spot": 
    "This disease causes small spots on leaves leading to defoliation. Remove infected leaves and apply fungicides regularly. Avoid overhead watering and practice crop rotation to reduce infection.",

    "Tomato___Spider_mites Two-spotted_spider_mite": 
    "Spider mites are pests that suck plant sap. Use neem oil, insecticidal soap, or miticides to control them. Maintain proper humidity and wash plants with water spray to reduce mite population.",

    "Tomato___Target_Spot": 
    "Target spot appears as concentric rings on leaves. Remove infected leaves and apply fungicides. Ensure proper spacing and avoid excessive moisture to prevent disease spread.",

    "Tomato___Tomato_mosaic_virus": 
    "This viral disease spreads through contact. Remove and destroy infected plants immediately. Disinfect tools and avoid handling healthy plants after infected ones. Use resistant varieties.",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 
    "This virus is transmitted by whiteflies. Control whiteflies using insecticides or sticky traps. Remove infected plants and use resistant varieties. Maintain field hygiene to prevent spread."
}

# Preprocess image
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    suggestion = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Please choose an image to upload."
        else:
            try:
                img = Image.open(file)
                img = preprocess(img)
            except UnidentifiedImageError:
                error = "That file is not a valid image. Please upload a leaf photo."
            else:
                probs = model.predict(img)[0]
                idx = int(np.argmax(probs))

                prediction = class_names[idx]
                confidence = round(float(probs[idx]) * 100, 2)
                suggestion = suggestions.get(prediction, "No suggestion available.")

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        suggestion=suggestion,
        error=error
    )

if __name__ == "__main__":
    # Debug mode is opt-in; never enable the interactive debugger in production.
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug)
