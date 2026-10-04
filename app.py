import os
import re
import base64
import hashlib
import hmac
import secrets
import sqlite3
from urllib.parse import quote

import joblib
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="centered")

LABELS = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "No Tumor Detected",
    "pituitary": "Pituitary Tumor",
}

# ---------------------------------------------------------------
# Background (medical / tumor theme)
# Put a file named background.jpg next to app.py to use your own image.
# ---------------------------------------------------------------
BRAIN_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 800' preserveAspectRatio='xMidYMid slice'>
  <defs>
    <radialGradient id='t' cx='50%' cy='50%' r='50%'>
      <stop offset='0%' stop-color='#ff5a4d' stop-opacity='0.95'/>
      <stop offset='60%' stop-color='#ff5a4d' stop-opacity='0.25'/>
      <stop offset='100%' stop-color='#ff5a4d' stop-opacity='0'/>
    </radialGradient>
    <radialGradient id='g' cx='50%' cy='45%' r='60%'>
      <stop offset='0%' stop-color='#1b6ca8' stop-opacity='0.55'/>
      <stop offset='100%' stop-color='#0a1a33' stop-opacity='0'/>
    </radialGradient>
  </defs>
  <rect width='1200' height='800' fill='#07152b'/>
  <ellipse cx='600' cy='380' rx='520' ry='360' fill='url(#g)'/>
  <g fill='none' stroke='#5fd4ff' stroke-opacity='0.45' stroke-width='3' stroke-linecap='round'>
    <path d='M600 150 C480 120 340 170 320 290 C270 310 250 400 300 450 C290 520 360 580 440 570 C480 620 560 640 620 610 C700 650 790 620 830 560 C920 560 980 480 940 410 C990 350 950 260 880 240 C860 160 720 110 600 150 Z'/>
    <path d='M600 150 C590 260 610 330 600 420 C590 500 610 560 620 610'/>
    <path d='M380 250 C430 230 470 270 450 310 C430 350 380 330 400 380'/>
    <path d='M470 200 C520 220 540 260 510 300 C480 340 520 380 560 360'/>
    <path d='M360 430 C410 410 450 450 430 490 C410 530 460 540 500 520'/>
    <path d='M700 190 C740 210 760 250 730 290 C700 330 740 360 780 340'/>
    <path d='M840 280 C880 300 890 350 860 380 C830 410 870 440 910 430'/>
    <path d='M690 470 C730 450 780 480 760 520 C740 560 790 570 820 540'/>
    <path d='M540 440 C580 470 560 520 520 540 C490 555 500 590 540 590'/>
  </g>
  <circle cx='770' cy='335' r='95' fill='url(#t)'/>
  <circle cx='770' cy='335' r='34' fill='#ff6b5e' fill-opacity='0.85'/>
  <circle cx='770' cy='335' r='62' fill='none' stroke='#ffb3ab' stroke-width='2.5' stroke-dasharray='7 7'/>
  <g stroke='#7fe3ff' stroke-opacity='0.35' stroke-width='6' stroke-linecap='round'>
    <path d='M90 120h36M108 102v36'/><path d='M1040 90h36M1058 72v36'/>
    <path d='M170 640h36M188 622v36'/><path d='M1010 690h36M1028 672v36'/>
    <path d='M1120 420h28M1134 406v28'/><path d='M60 380h28M74 366v28'/>
  </g>
  <polyline fill='none' stroke='#37e6a6' stroke-opacity='0.5' stroke-width='3'
    points='0,740 180,740 220,740 250,690 285,775 320,715 350,740 560,740 600,740 630,700 660,770 690,725 715,740 1200,740'/>
  <g fill='#7fe3ff' fill-opacity='0.5'>
    <circle cx='200' cy='250' r='4'/><circle cx='260' cy='210' r='3'/><circle cx='1000' cy='180' r='4'/>
    <circle cx='1060' cy='250' r='3'/><circle cx='980' cy='620' r='4'/><circle cx='230' cy='560' r='3'/>
  </g>
</svg>
"""


def set_background():
    if os.path.exists("background.jpg"):
        with open("background.jpg", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        bg = f'url("data:image/jpeg;base64,{b64}")'
    else:
        bg = f'url("data:image/svg+xml;utf8,{quote(BRAIN_SVG)}")'

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(5,14,30,0.55), rgba(5,14,30,0.80)), {bg};
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        header[data-testid="stHeader"] {{ background: transparent; }}
        .block-container {{
            background: rgba(10, 25, 50, 0.72);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(95, 212, 255, 0.25);
            border-radius: 18px;
            padding: 2rem 2rem 2.5rem 2rem;
            margin-top: 2rem;
            max-width: 760px;
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp p,
        .stApp [data-testid="stMarkdownContainer"] {{ color: #e8f4ff; }}
        .brand {{ text-align:center; margin-bottom: .2rem; }}
        .brand h1 {{ font-size: 2.2rem; margin-bottom: 0; }}
        .tagline {{ text-align:center; color:#8fd8ff; margin-bottom: 1.2rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------
# User accounts (SQLite + salted PBKDF2 password hashing)
# ---------------------------------------------------------------
DB_PATH = "users.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
               username TEXT PRIMARY KEY,
               email TEXT UNIQUE NOT NULL,
               full_name TEXT NOT NULL,
               salt TEXT NOT NULL,
               pw_hash TEXT NOT NULL)"""
    )
    return conn


def hash_pw(password, salt_hex):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt_hex), 200_000
    ).hex()


def create_user(full_name, username, email, password):
    salt = secrets.token_hex(16)
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO users VALUES (?,?,?,?,?)",
                (username.lower(), email.lower(), full_name, salt, hash_pw(password, salt)),
            )
        return True, "Account created! Please sign in."
    except sqlite3.IntegrityError:
        return False, "That username or email is already registered."


def verify_user(username, password):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT salt, pw_hash, full_name FROM users WHERE username=?",
            (username.lower(),),
        ).fetchone()
    if row and hmac.compare_digest(hash_pw(password, row[0]), row[1]):
        return row[2]
    return None


# ---------------------------------------------------------------
# Sign in / Sign up page
# ---------------------------------------------------------------
def auth_page():
    st.markdown('<div class="brand"><h1>🧠 NeuroScan AI</h1></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tagline">Brain tumor MRI classification · CNN + ZOA + SVM</div>',
        unsafe_allow_html=True,
    )
    tab_in, tab_up = st.tabs(["🔐 Sign In", "📝 Sign Up"])

    with tab_in:
        with st.form("signin"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            go = st.form_submit_button("Sign In", use_container_width=True)
        if go:
            name = verify_user(u.strip(), p) if u.strip() and p else None
            if name:
                st.session_state["user"] = name
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab_up:
        with st.form("signup"):
            full = st.text_input("Full name")
            user = st.text_input("Choose a username")
            mail = st.text_input("Email")
            p1 = st.text_input("Password (min 6 characters)", type="password")
            p2 = st.text_input("Confirm password", type="password")
            go = st.form_submit_button("Create Account", use_container_width=True)
        if go:
            if not (full.strip() and user.strip() and mail.strip() and p1):
                st.error("Please fill in all fields.")
            elif not re.fullmatch(r"[A-Za-z0-9_]{3,20}", user.strip()):
                st.error("Username must be 3-20 letters, numbers or underscores.")
            elif not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", mail.strip()):
                st.error("Please enter a valid email address.")
            elif len(p1) < 6:
                st.error("Password must be at least 6 characters.")
            elif p1 != p2:
                st.error("Passwords do not match.")
            else:
                ok, msg = create_user(full.strip(), user.strip(), mail.strip(), p1)
                (st.success if ok else st.error)(msg)

    st.caption("For research and educational use only. Not a substitute for professional medical diagnosis.")


# ---------------------------------------------------------------
# Classifier page (shown after login)
# ---------------------------------------------------------------
@st.cache_resource
def load_models():
    from tensorflow.keras.applications import MobileNetV2

    path = "model/zoa_svm.joblib" if os.path.exists("model/zoa_svm.joblib") else "zoa_svm.joblib"
    bundle = joblib.load(path)
    cnn = MobileNetV2(weights="imagenet", include_top=False,
                      pooling="avg", input_shape=(224, 224, 3))
    return bundle, cnn


def classifier_page():
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    with st.sidebar:
        st.markdown(f"### 👋 Welcome, {st.session_state['user']}")
        if st.button("Log out", use_container_width=True):
            del st.session_state["user"]
            st.rerun()

    st.markdown('<div class="brand"><h1>🧠 NeuroScan AI</h1></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tagline">Upload a brain MRI image for automated tumor classification</div>',
        unsafe_allow_html=True,
    )

    bundle, cnn = load_models()
    scaler, mask, svm, classes = (bundle["scaler"], bundle["mask"],
                                  bundle["svm"], bundle["classes"])

    uploaded = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded MRI", use_container_width=True)

        if st.button("Analyze", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                arr = preprocess_input(
                    np.array(img.resize((224, 224)), dtype="float32")[None, ...])
                feats = cnn.predict(arr, verbose=0)       # CNN features (1280)
                X = scaler.transform(feats)[:, mask]      # ZOA-selected features
                proba = svm.predict_proba(X)[0]           # SVM classification
            k = int(proba.argmax())

            st.subheader(LABELS.get(classes[k], classes[k]))
            st.write(f"Confidence: **{proba[k] * 100:.2f}%**")
            for c, p in zip(classes, proba):
                st.write(f"{LABELS.get(c, c)} — {p * 100:.1f}%")
                st.progress(float(p))
            st.caption(f"ZOA selected {int(mask.sum())} of {len(mask)} CNN features.")

    st.warning("For research and educational use only. Not a substitute for professional medical diagnosis.")


# ---------------------------------------------------------------
set_background()
if "user" in st.session_state:
    classifier_page()
else:
    auth_page()
