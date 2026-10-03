import io
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024   # 8 MB upload limit

bundle = joblib.load("model/zoa_svm.joblib")
scaler, mask, svm, classes = (bundle["scaler"], bundle["mask"],
                              bundle["svm"], bundle["classes"])

# Stage 1: CNN feature extractor
cnn = MobileNetV2(weights="imagenet", include_top=False,
                  pooling="avg", input_shape=(224, 224, 3))

LABELS = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "No Tumor Detected",
    "pituitary": "Pituitary Tumor",
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="No image uploaded"), 400
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB").resize((224, 224))
    except Exception:
        return jsonify(error="Invalid image file"), 400

    arr = preprocess_input(np.array(img, dtype="float32")[None, ...])
    feats = cnn.predict(arr, verbose=0)               # CNN features (1280)
    X = scaler.transform(feats)[:, mask]              # ZOA-selected features
    proba = svm.predict_proba(X)[0]                   # SVM classification
    k = int(proba.argmax())

    return jsonify(
        label=LABELS.get(classes[k], classes[k]),
        confidence=round(float(proba[k]) * 100, 2),
        probabilities={LABELS.get(c, c): round(float(p) * 100, 2)
                       for c, p in zip(classes, proba)},
        features_used=int(mask.sum()),
        features_total=int(len(mask)),
    )

if __name__ == "__main__":
    app.run(debug=True)
