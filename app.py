import joblib
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠")

LABELS = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "No Tumor Detected",
    "pituitary": "Pituitary Tumor",
}

@st.cache_resource
def load_models():
    bundle = joblib.load("model/zoa_svm.joblib")
    cnn = MobileNetV2(weights="imagenet", include_top=False,
                      pooling="avg", input_shape=(224, 224, 3))
    return bundle, cnn

bundle, cnn = load_models()
scaler, mask, svm, classes = (bundle["scaler"], bundle["mask"],
                              bundle["svm"], bundle["classes"])

st.title("🧠 Brain Tumor Classification")
st.caption("Hybrid CNN + Zebra Optimization Algorithm (ZOA) + SVM")

uploaded = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded MRI", use_container_width=True)

    if st.button("Analyze", type="primary"):
        with st.spinner("Analyzing..."):
            arr = preprocess_input(np.array(img.resize((224, 224)), dtype="float32")[None, ...])
            feats = cnn.predict(arr, verbose=0)          # CNN features (1280)
            X = scaler.transform(feats)[:, mask]         # ZOA-selected features
            proba = svm.predict_proba(X)[0]              # SVM classification
        k = int(proba.argmax())

        st.subheader(LABELS.get(classes[k], classes[k]))
        st.write(f"Confidence: **{proba[k] * 100:.2f}%**")
        for c, p in zip(classes, proba):
            st.write(LABELS.get(c, c))
            st.progress(float(p))
        st.caption(f"ZOA selected {int(mask.sum())} of {len(mask)} CNN features.")

st.warning("For research and educational use only. Not a substitute for professional medical diagnosis.")
