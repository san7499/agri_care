import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask app
app = Flask(__name__)

# Temporary upload folder
UPLOAD_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "agricare.keras")
model = load_model(MODEL_PATH)

# Class labels
class_labels = [
    'Pepper bell Bacterial spot', 'Pepper bell healthy',
    'Potato Early blight', 'Potato Late blight', 'Potato healthy',
    'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight',
    'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
    'Tomato Spider mites Two spotted spider mite', 'Tomato Target Spot',
    'Tomato Tomato YellowLeaf Curl Virus', 'Tomato Tomato mosaic virus',
    'Tomato healthy'
]

# Treatment suggestions
treatment_suggestions = {
    'Pepper bell Bacterial spot': {
        'Fertilizer': ['Balanced NPK fertilizer', 'Potassium-rich fertilizer for stronger stems'],
        'Pesticide': ['Copper-based fungicides (Bordeaux mixture)', 'Mancozeb spray'],
        'Organic': ['Neem oil spray', 'Compost tea']
    },
    'Pepper bell healthy': {
        'Fertilizer': ['Balanced NPK fertilizer', 'Organic compost'],
        'Pesticide': ['No chemical pesticide required'],
        'Organic': ['Mulching', 'Compost application']
    },
    'Potato Early blight': {
        'Fertilizer': ['Potassium-rich fertilizer', 'Balanced NPK with micronutrients'],
        'Pesticide': ['Mancozeb fungicide', 'Chlorothalonil spray'],
        'Organic': ['Neem extract sprays', 'Crop rotation']
    },
    'Potato Late blight': {
        'Fertilizer': ['Balanced NPK fertilizer with phosphorus', 'Calcium-rich fertilizer'],
        'Pesticide': ['Metalaxyl-based fungicides', 'Copper fungicides'],
        'Organic': ['Resistant varieties', 'Proper drainage and compost']
    },
    'Potato healthy': {
        'Fertilizer': ['Balanced NPK fertilizer', 'Compost-enriched soil'],
        'Pesticide': ['No chemical pesticide needed'],
        'Organic': ['Apply compost', 'Maintain good irrigation']
    },
    'Tomato Bacterial spot': {
        'Fertilizer': ['Balanced NPK fertilizer', 'Nitrogen-rich fertilizer for leaf growth'],
        'Pesticide': ['Copper sprays', 'Bacillus subtilis-based biopesticides'],
        'Organic': ['Neem oil sprays', 'Remove infected leaves']
    },
    'Tomato Early blight': {
        'Fertilizer': ['Compost-enriched fertilizer', 'Potassium-rich fertilizer'],
        'Pesticide': ['Chlorothalonil', 'Copper fungicides'],
        'Organic': ['Neem extract', 'Compost tea sprays']
    },
    'Tomato Late blight': {
        'Fertilizer': ['Phosphorus-rich fertilizer', 'Balanced NPK fertilizer'],
        'Pesticide': ['Metalaxyl sprays', 'Copper-based fungicides'],
        'Organic': ['Copper sprays', 'Ensure proper drainage']
    },
    'Tomato Leaf Mold': {
        'Fertilizer': ['Balanced NPK fertilizer', 'Nitrogen for leaf growth'],
        'Pesticide': ['Chlorothalonil', 'Copper fungicides'],
        'Organic': ['Neem oil sprays', 'Improve ventilation']
    },
    'Tomato Septoria leaf spot': {
        'Fertilizer': ['Potassium-rich fertilizer', 'Balanced NPK fertilizer'],
        'Pesticide': ['Mancozeb fungicide', 'Copper sprays'],
        'Organic': ['Remove infected leaves', 'Compost tea application']
    },
    'Tomato Spider mites Two spotted spider mite': {
        'Fertilizer': ['Balanced NPK fertilizer', 'Nitrogen fertilizer'],
        'Pesticide': ['Miticides', 'Acaricides'],
        'Organic': ['Neem oil', 'Insecticidal soap']
    },
    'Tomato Target Spot': {
        'Fertilizer': ['Balanced fertilizer', 'Potassium-enriched fertilizer'],
        'Pesticide': ['Copper sprays', 'Mancozeb sprays'],
        'Organic': ['Neem oil sprays', 'Remove affected leaves']
    },
    'Tomato Tomato YellowLeaf Curl Virus': {
        'Fertilizer': ['Balanced fertilizer', 'Nitrogen-phosphorus-potassium fertilizer'],
        'Pesticide': ['Control whiteflies with insecticides', 'Neem-based biopesticides'],
        'Organic': ['Yellow sticky traps', 'Neem oil for vectors']
    },
    'Tomato Tomato mosaic virus': {
        'Fertilizer': ['Potassium-rich fertilizer', 'Balanced NPK fertilizer'],
        'Pesticide': ['No chemical cure', 'Control vectors manually'],
        'Organic': ['Remove infected plants', 'Use resistant varieties']
    },
    'Tomato healthy': {
        'Fertilizer': ['Balanced NPK fertilizer', 'Compost application'],
        'Pesticide': ['No pesticide needed'],
        'Organic': ['Compost', 'Maintain irrigation']
    }
}

# Prediction function
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)

    predicted_class = class_labels[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    treatment = treatment_suggestions.get(predicted_class, {
        'Fertilizer': ["General balanced fertilizer"],
        'Pesticide': ["No chemical needed"],
        'Organic': ["Maintain compost and irrigation"]
    })

    return predicted_class, confidence, treatment

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"
        file = request.files["file"]
        if file.filename == "":
            return "No image selected"

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        prediction, confidence, treatment = predict_disease(filepath)
        return render_template(
            "index.html",
            file=file.filename,
            prediction=prediction,
            confidence=confidence,
            fertilizer=treatment['Fertilizer'],
            pesticide=treatment['Pesticide'],
            organic=treatment['Organic']
        )
    return render_template("index.html")

# Optional route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
