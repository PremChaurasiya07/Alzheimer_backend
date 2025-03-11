from flask import Flask, request, jsonify
import joblib
import numpy as np
import cv2
import os
import onnxruntime as ort
from werkzeug.utils import secure_filename
import pandas as pd

app = Flask(__name__)


random_forest_model = joblib.load("Alzheimer_Model.pkl")  # For structured data
onnx_session = ort.InferenceSession("Mri_Alzhiemer.onnx")  # For MRI images


le_gender = joblib.load("./encoder_model1/le_gender.pkl")
le_hand = joblib.load("./encoder_model1/le_hand.pkl")


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    """Check if file has a valid image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for the ONNX model."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0  
    image = np.expand_dims(image, axis=0)  

    return image

def normalize_gender(value):
    """Normalize gender input."""
    value = str(value).strip().lower()
    return "M" if value in ["m", "male"] else "F" if value in ["f", "female"] else None

def normalize_hand(value):
    """Normalize hand input."""
    value = str(value).strip().lower()
    return "R" if value in ["r", "right", "right-handed"] else "L" if value in ["l", "left", "left-handed"] else None

def safe_transform(le, value, feature_name):
    """Safely transform categorical values to numerical labels."""
    try:
        return le.transform([value])[0]
    except ValueError:
        print(f"Warning: Unseen label '{value}' in feature '{feature_name}'")
        return -1  


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

      
        df["M/F"] = df["M/F"].apply(normalize_gender)
        df["Hand"] = df["Hand"].apply(normalize_hand)

        if df["M/F"].isnull().any() or df["Hand"].isnull().any():
            return jsonify({"error": "Invalid values for M/F or Hand. Use 'male/female' and 'right/left'."})

       
        df["M/F"] = df["M/F"].apply(lambda x: safe_transform(le_gender, x, "M/F"))
        df["Hand"] = df["Hand"].apply(lambda x: safe_transform(le_hand, x, "Hand"))

        
        prediction = random_forest_model.predict(df).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})



@app.route("/predict-mri", methods=["POST"])
def predict_mri():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"})
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"})

        if file and allowed_file(file.filename):
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
        
        return jsonify({"error": "Invalid file format. Allowed: png, jpg, jpeg."})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)  
app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))

