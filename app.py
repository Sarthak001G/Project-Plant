from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json

app = Flask(__name__)

# Load model
model = load_model("model/plant_model.keras")

# Load class names
with open("model/class_names.json", "r") as f:
    class_names = json.load(f)

def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            img = Image.open(file)
            img = preprocess(img)

            pred = model.predict(img)
            prediction = class_names[np.argmax(pred)]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
#     app.run(debug=True)from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
# import json
# import os

# # Data generators
# train_gen = ImageDataGenerator(rescale=1./255)
# val_gen = ImageDataGenerator(rescale=1./255)

# train_data = train_gen.flow_from_directory(
#     r"C:\Users\sarth\OneDrive\Desktop\project plant\Dataset\train",
#     target_size=(224,224),
#     batch_size=32,
#     class_mode='categorical'
# )

# val_data = val_gen.flow_from_directory(
#     r"C:\Users\sarth\OneDrive\Desktop\project plant\Dataset\valid",
#     target_size=(224,224),
#     batch_size=32,
#     class_mode='categorical'
# )

# # ✅ Save class names (VERY IMPORTANT)
# class_names = list(train_data.class_indices.keys())
# os.makedirs("model", exist_ok=True)

# with open("model/class_names.json", "w") as f:
#     json.dump(class_names, f)

# # Model
# model = Sequential([
#     Input(shape=(224,224,3)),

#     Conv2D(32, (3,3), activation='relu'),
#     MaxPooling2D(),

#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(),

#     Conv2D(128, (3,3), activation='relu'),
#     MaxPooling2D(),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),

#     Dense(train_data.num_classes, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(train_data, epochs=5, validation_data=val_data)  

# # ✅ Save in new format
# model.save("model/plant_model.keras")

# print("✅ Model + class names saved!")<!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <title>Plant Disease Detector 🌿</title>
#     <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
# </head>
# <body>

# <div class="container">

#     <h1 class="title">🌿 Plant Disease Detection</h1>
#     <p class="subtitle">Upload a leaf image to detect disease instantly</p>

#     <form method="POST" enctype="multipart/form-data" class="upload-box">
        
#         <input type="file" name="image" id="fileInput" required>
        
#         <label for="fileInput" class="custom-file">
#             📁 Choose Image
#         </label>

#         <button type="submit" class="btn">🔍 Predict</button>
#     </form>

#     {% if prediction %}
#     <div class="result-box">
#         <h2>Prediction:</h2>
#         <p class="result">{{ prediction }}</p>
#     </div>
#     {% endif %}

# </div>

# </body>
# </html>/* 🌈 Background Gradient Animation */
# body {
#     margin: 0;
#     font-family: 'Segoe UI', sans-serif;
#     background: linear-gradient(-45deg, #1b5e20, #43a047, #66bb6a, #a5d6a7);
#     background-size: 400% 400%;
#     animation: gradientBG 10s ease infinite;
#     color: white;
#     text-align: center;
# }

# /* 🌿 Container */
# .container {
#     margin-top: 80px;
#     animation: fadeIn 1.5s ease;
# }

# /* ✨ Title */
# .title {
#     font-size: 40px;
#     font-weight: bold;
#     animation: slideDown 1s ease;
# }

# .subtitle {
#     margin-bottom: 30px;
#     font-size: 18px;
#     opacity: 0.9;
# }

# /* 📦 Upload Box */
# .upload-box {
#     display: inline-block;
#     padding: 30px;
#     border-radius: 15px;
#     background: rgba(255,255,255,0.1);
#     backdrop-filter: blur(10px);
#     box-shadow: 0 8px 32px rgba(0,0,0,0.2);
#     animation: fadeInUp 1.2s ease;
# }

# /* Hide default input */
# input[type="file"] {
#     display: none;
# }

# /* Custom file button */
# .custom-file {
#     display: inline-block;
#     padding: 12px 20px;
#     background: white;
#     color: #2e7d32;
#     border-radius: 8px;
#     cursor: pointer;
#     margin-bottom: 15px;
#     transition: 0.3s;
# }

# .custom-file:hover {
#     background: #c8e6c9;
# }

# /* Predict button */
# .btn {
#     display: block;
#     width: 100%;
#     padding: 12px;
#     background: #00c853;
#     border: none;
#     border-radius: 8px;
#     color: white;
#     font-size: 16px;
#     cursor: pointer;
#     transition: 0.3s;
# }

# .btn:hover {
#     background: #00e676;
#     transform: scale(1.05);
# }

# /* 📊 Result Box */
# .result-box {
#     margin-top: 30px;
#     padding: 20px;
#     border-radius: 12px;
#     background: rgba(0,0,0,0.3);
#     animation: popUp 0.6s ease;
# }

# .result {
#     font-size: 22px;
#     font-weight: bold;
#     color: #a5d6a7;
# }

# /* 🔥 Animations */
# @keyframes gradientBG {
#     0% { background-position: 0% 50%; }
#     50% { background-position: 100% 50%; }
#     100% { background-position: 0% 50%; }
# }

# @keyframes fadeIn {
#     from {opacity: 0;}
#     to {opacity: 1;}
# }

# @keyframes fadeInUp {
#     from {
#         transform: translateY(40px);
#         opacity: 0;
#     }
#     to {
#         transform: translateY(0);
#         opacity: 1;
#     }
# }

# @keyframes slideDown {
#     from {
#         transform: translateY(-50px);
#         opacity: 0;
#     }
#     to {
#         transform: translateY(0);
#         opacity: 1;
#     }
# }

# @keyframes popUp {
#     from {
#         transform: scale(0.8);
#         opacity: 0;
#     }
#     to {
#         transform: scale(1);
#         opacity: 1;
#     }
# }







# # from flask import Flask, render_template, request
# # from tensorflow.keras.models import load_model
# # from PIL import Image
# # import numpy as np

# # app = Flask(__name__)

# # model = load_model("model/plant_model.h5")

# # class_names = list(train_data.class_indices.keys())

# # def preprocess(img):
# #     img = img.resize((224,224))
# #     img = np.array(img)/255.0
# #     img = np.expand_dims(img, axis=0)
# #     return img

# # @app.route("/", methods=["GET", "POST"])
# # def index():
# #     prediction = ""

# #     if request.method == "POST":
# #         file = request.files["image"]
# #         img = Image.open(file)

# #         img = preprocess(img)
# #         pred = model.predict(img)

# #         prediction = class_names[np.argmax(pred)]

# #     return render_template("index.html", prediction=prediction)

# # if __name__ == "__main__":
# #     app.run(debug=True)