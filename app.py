import os
import cv2
import numpy as np
import joblib
import pandas as pd
import onnxruntime as ort
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Import CORS

# Initialize Flask app
app = Flask(__name__)

# Apply CORS to all routes
CORS(app)

# If you need to customize CORS (like allowing only specific domains)
# CORS(app, resources={r"/predict": {"origins": "http://example.com"}})

try:
    random_forest_model = joblib.load("Alzheimer_Model.pkl")  # Text model
    onnx_session = ort.InferenceSession("Mri_Alzhiemer.onnx")  # MRI model
    le_gender = joblib.load("./encoder_model1/le_gender.pkl")
    le_hand = joblib.load("./encoder_model1/le_hand.pkl")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for ONNX model input."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Invalid image file. Cannot read.")

        image = cv2.resize(image, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0  
        image = np.expand_dims(image, axis=0)  
        return image
    except Exception as e:
        raise ValueError(f"Error in image preprocessing: {e}")

def normalize_gender(value):
    value = str(value).strip().lower()
    return "M" if value in ["m", "male"] else "F" if value in ["f", "female"] else None

def normalize_hand(value):
    value = str(value).strip().lower()
    return "R" if value in ["r", "right", "right-handed"] else "L" if value in ["l", "left", "left-handed"] else None

def safe_transform(le, value, feature_name):
    try:
        return le.transform([value])[0]
    except ValueError:
        print(f"Warning: Unseen label '{value}' in feature '{feature_name}'")
        return -1  

@app.route("/")
def home():
    return jsonify({"message": "Alzheimer Prediction API is running!", "status": "success"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        df = pd.DataFrame([data])

        df["M/F"] = df["M/F"].apply(normalize_gender)
        df["Hand"] = df["Hand"].apply(normalize_hand)

        if df["M/F"].isnull().any() or df["Hand"].isnull().any():
            return jsonify({"error": "Invalid values for M/F or Hand. Use 'male/female' and 'right/left'."}), 400

        df["M/F"] = df["M/F"].apply(lambda x: safe_transform(le_gender, x, "M/F"))
        df["Hand"] = df["Hand"].apply(lambda x: safe_transform(le_hand, x, "Hand"))

        prediction = random_forest_model.predict(df).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict-mri", methods=["POST"])
def predict_mri():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            os.makedirs("uploads", exist_ok=True)

            filename = secure_filename(file.filename)
            file_path = os.path.join("uploads", filename)
            file.save(file_path) 

            image = preprocess_image(file_path)
            input_tensor = {onnx_session.get_inputs()[0].name: image}

            prediction = onnx_session.run(None, input_tensor)[0]
            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction))

            os.remove(file_path)

            return jsonify({"prediction": int(predicted_class), "confidence": confidence})
        
        return jsonify({"error": "Invalid file format. Allowed: png, jpg, jpeg."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)  
    port = int(os.getenv("PORT", 10000))  
    app.run(host="0.0.0.0", port=port)
