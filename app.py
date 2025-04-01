import os
import cv2
import numpy as np
import joblib
import pandas as pd
import onnxruntime as ort
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Import CORS
import base64  # Ensure this is imported at the top of the file
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt  # Ensure this is imported at the top of the file

# Initialize Flask app
app = Flask(__name__)

# Apply CORS to all routes
CORS(app,origins='*')

# If you need to customize CORS (like allowing only specific domains)
# CORS(app, resources={r"/predict": {"origins": "http://example.com"}})

try:
    random_forest_model = joblib.load("Alzheimer_Model.pkl")  # Text model
    onnx_session = ort.InferenceSession("fresh_alzheimer.onnx")  # MRI model
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        image = image.astype(np.float32)
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
                input_tensor = {onnx_session.get_inputs()[0].name: image}
                prediction = onnx_session.run(None, input_tensor)[0]
                confidence = float(np.max(prediction) * 100)  # Convert to percentage
                predicted_class = int(np.argmax(prediction))

                # Define class labels
                class_labels = {0: "No Dementia", 1: "Very Mild Dementia", 2: "Mild Dementia", 3: "Moderate Dementia"}
                predicted_label = class_labels.get(predicted_class, "Unknown")

                # Load the original image for visualization
                original_image = cv2.imread(file_path)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

                # Use matplotlib to overlay prediction text on the image
                plt.figure(figsize=(6, 6))
                plt.imshow(original_image)
                plt.axis("off")
                plt.title(f"{predicted_label} ({confidence:.2f}%)", fontsize=16, color="green")

                # Save the visualized image to a buffer with padding
                buffer = BytesIO()
                plt.savefig(buffer, format="JPEG", bbox_inches="tight", pad_inches=0.5)  # Add padding with pad_inches
                buffer.seek(0)

                # Convert the visualized image to Base64
                img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
                buffer.close()

                return jsonify({
                    "prediction": predicted_class,
                    "confidence": confidence,
                    "visualized_image_base64": img_base64  # Send Base64-encoded visualized image
                })
            finally:
                os.remove(file_path)  # Ensure file is deleted
        
        return jsonify({"error": "Invalid file format. Allowed: png, jpg, jpeg."}), 400
    except Exception as e:
        print(f"Error in /predict-mri: {e}")  # Log the error
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)  
    port = int(os.getenv("PORT", 10000))  
    app.run(host="0.0.0.0", port=port)
