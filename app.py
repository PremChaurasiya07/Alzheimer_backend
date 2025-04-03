import os
import cv2
import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Import CORS
import onnxruntime as ort  # Import ONNX Runtime

# Initialize Flask app
app = Flask(__name__)

# Apply CORS to all routes
CORS(app, origins='*')

# Load models
try:
    random_forest_model = joblib.load("Alzheimer_Model.pkl")  # Text model
    onnx_session = ort.InferenceSession("new_Alzheimer.onnx")  # Load ONNX model
    le_gender = joblib.load("./encoder_model1/le_gender.pkl")
    le_hand = joblib.load("./encoder_model1/le_hand.pkl")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess an MRI image for ONNX model input."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Invalid image file. Cannot read.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))  # Ensure this matches the ONNX model input size
        image = image.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension
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
    print(request.form)  # Log form data
    print(request.files)
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

            try:
                # Preprocess the image
                image = preprocess_image(file_path)

                # Run the ONNX model
                input_name = onnx_session.get_inputs()[0].name
                prediction = onnx_session.run(None, {input_name: image})[0]
                confidence = float(np.max(prediction) * 100)  # Convert to percentage
                predicted_class = int(np.argmax(prediction))

                # Define class labels
                class_labels = {0: "Mild Demented", 1: "Moderate Demented", 2: "Non Demented", 3: "Very Mild Dementia"}
                predicted_label = class_labels.get(predicted_class, "Unknown")

                os.remove(file_path)  # Clean up the uploaded file

                return jsonify({
                    "prediction": predicted_label,
                    "prediction_num": predicted_class,
                    "probabilities": prediction.tolist(),
                    "confidence": confidence
                })
            except Exception as e:
                os.remove(file_path)  # Ensure file is deleted in case of an error
                raise e
        
        return jsonify({"error": "Invalid file format. Allowed: png, jpg, jpeg."}), 400
    except Exception as e:
        print(f"Error in /predict-mri: {e}")  # Log the error
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)  
    port = int(os.getenv("PORT", 10000))  
    app.run(host="0.0.0.0", port=port)
