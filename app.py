import io
import os
import random
import re
import time
import zipfile
from datetime import datetime, timedelta, timezone
import base64
import hashlib
import hmac
import secrets
import sqlite3
from urllib.parse import quote

import joblib
import numpy as np
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="MRI NeuroScan AI", page_icon="🧠", layout="centered")

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
# Storage: Supabase (permanent) when configured, otherwise SQLite (temporary)
# Accounts + login/usage events are stored here.
# ---------------------------------------------------------------
DB_PATH = "users.db"


def _secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


SB_URL = str(_secret("SUPABASE_URL") or "").strip().rstrip("/")
if SB_URL.endswith("/rest/v1"):
    SB_URL = SB_URL[: -len("/rest/v1")]
SB_KEY = str(_secret("SUPABASE_KEY") or "").strip()
USE_SUPABASE = bool(SB_URL and SB_KEY)


def _sb(method, table, params=None, json=None, headers=None):
    h = {"apikey": SB_KEY, "Content-Type": "application/json"}
    if SB_KEY.startswith("eyJ"):                 # legacy JWT-style keys also need Authorization
        h["Authorization"] = f"Bearer {SB_KEY}"
    if headers:
        h.update(headers)
    return requests.request(method, f"{SB_URL}/rest/v1/{table}",
                            params=params, json=json, headers=h, timeout=15)


def _sb_count(table, filters=None):
    params = {"select": "username", "limit": "1"}
    if filters:
        params.update(filters)
    r = _sb("GET", table, params=params, headers={"Prefer": "count=exact"})
    r.raise_for_status()
    return int(r.headers.get("Content-Range", "*/0").split("/")[-1])


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
               username TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
               full_name TEXT NOT NULL, salt TEXT NOT NULL, pw_hash TEXT NOT NULL,
               created_at TEXT DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS events (
               id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, event TEXT,
               detail TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"""
    )
    return conn


def store_create_user(full_name, username, email, salt, pw_hash):
    if USE_SUPABASE:
        r = _sb("POST", "users", headers={"Prefer": "return=minimal"},
                json={"username": username, "email": email, "full_name": full_name,
                      "salt": salt, "pw_hash": pw_hash})
        if r.status_code in (200, 201, 204):
            return True, ""
        if r.status_code == 409:
            return False, "That username or email is already registered."
        return False, f"Database error ({r.status_code}): {r.text[:200]}"
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO users (username,email,full_name,salt,pw_hash) VALUES (?,?,?,?,?)",
                (username, email, full_name, salt, pw_hash))
        return True, ""
    except sqlite3.IntegrityError:
        return False, "That username or email is already registered."


def store_find_user(ident):
    if USE_SUPABASE:
        for col in ("username", "email"):
            r = _sb("GET", "users", params={col: f"eq.{ident}", "limit": "1",
                                            "select": "username,full_name,salt,pw_hash"})
            r.raise_for_status()
            data = r.json()
            if data:
                return data[0]
        return None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT username, full_name, salt, pw_hash FROM users WHERE username=? OR email=?",
            (ident, ident)).fetchone()
    return dict(zip(("username", "full_name", "salt", "pw_hash"), row)) if row else None


def store_log(username, event, detail=""):
    """Record signup / login / login_failed / analysis events (never breaks the app)."""
    try:
        if USE_SUPABASE:
            _sb("POST", "events", headers={"Prefer": "return=minimal"},
                json={"username": username, "event": event, "detail": detail[:200]})
        else:
            with get_conn() as conn:
                conn.execute("INSERT INTO events (username,event,detail) VALUES (?,?,?)",
                             (username, event, detail[:200]))
    except Exception:
        pass


def store_stats():
    if USE_SUPABASE:
        r = _sb("GET", "events", params={"select": "username", "event": "eq.login", "limit": "10000"})
        r.raise_for_status()
        return {
            "users": _sb_count("users"),
            "logins": _sb_count("events", {"event": "eq.login"}),
            "unique_logins": len({x["username"] for x in r.json()}),
            "analyses": _sb_count("events", {"event": "eq.analysis"}),
        }
    with get_conn() as c:
        return {
            "users": c.execute("SELECT COUNT(*) FROM users").fetchone()[0],
            "logins": c.execute("SELECT COUNT(*) FROM events WHERE event='login'").fetchone()[0],
            "unique_logins": c.execute(
                "SELECT COUNT(DISTINCT username) FROM events WHERE event='login'").fetchone()[0],
            "analyses": c.execute("SELECT COUNT(*) FROM events WHERE event='analysis'").fetchone()[0],
        }


def store_recent(limit=100):
    if USE_SUPABASE:
        r = _sb("GET", "events", params={"select": "created_at,username,event,detail",
                                         "order": "created_at.desc", "limit": str(limit)})
        r.raise_for_status()
        return r.json()
    with get_conn() as c:
        rows = c.execute("SELECT created_at, username, event, detail FROM events "
                         "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(zip(("created_at", "username", "event", "detail"), x)) for x in rows]


def hash_pw(password, salt_hex):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt_hex), 200_000
    ).hex()


def create_user(full_name, username, email, password):
    salt = secrets.token_hex(16)
    try:
        ok, msg = store_create_user(full_name, username.lower(), email.lower(),
                                    salt, hash_pw(password, salt))
    except Exception:
        return False, "Could not reach the database. Please try again in a moment."
    if ok:
        store_log(username.lower(), "signup")
        return True, "Account created! Please sign in."
    return False, msg


def verify_user(identifier, password):
    """identifier = username OR email. Returns the user row if the password is right."""
    row = store_find_user(identifier.strip().lower())
    if row and hmac.compare_digest(hash_pw(password, row["salt"]), row["pw_hash"]):
        return row
    return None


# ---------------------------------------------------------------
# Sign in / Sign up page
# ---------------------------------------------------------------
def auth_page():
    st.markdown('<div class="brand"><h1>🧠 MRI NeuroScan AI</h1></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tagline">Brain tumor MRI classification · CNN + ZOA + SVM</div>',
        unsafe_allow_html=True,
    )
    if not USE_SUPABASE:
        st.caption("⚠ Temporary storage mode: accounts are erased whenever the app restarts.")
    tab_in, tab_up = st.tabs(["🔐 Sign In", "📝 Sign Up"])

    with tab_in:
        with st.form("signin"):
            u = st.text_input("Username or Email")
            p = st.text_input("Password", type="password")
            go = st.form_submit_button("Sign In", use_container_width=True)
        if go:
            try:
                row = verify_user(u, p) if u.strip() and p else None
            except Exception:
                st.error("Could not reach the database. Please try again in a moment.")
            else:
                if row:
                    store_log(row["username"], "login")
                    st.session_state["user"] = row["full_name"]
                    st.session_state["username"] = row["username"]
                    st.session_state["show_welcome"] = True
                    st.session_state["page"] = PAGE_DASH
                    st.rerun()
                else:
                    store_log(u.strip().lower()[:50], "login_failed")
                    st.error("Invalid username/email or password.")

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
# Model loading
# ---------------------------------------------------------------
@st.cache_resource
def load_models():
    from tensorflow.keras.applications import MobileNetV2

    path = "model/zoa_svm.joblib" if os.path.exists("model/zoa_svm.joblib") else "zoa_svm.joblib"
    bundle = joblib.load(path)
    cnn = MobileNetV2(weights="imagenet", include_top=False,
                      pooling="avg", input_shape=(224, 224, 3))
    return bundle, cnn


# ---------------------------------------------------------------
# Reading uploaded files: images, PDF, Word (.docx)
# ---------------------------------------------------------------
UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp", "pdf", "docx", "doc"]
FOV_MM = 220       # assumed field of view across the image width (mm)
MIN_SIDE = 64      # ignore tiny images such as logos/icons
MAX_IMAGES = 12


def images_from_file(uploaded):
    """Return a list of RGB PIL images found in the uploaded file."""
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    images = []

    if name.endswith(".pdf"):
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        for page in doc:
            for item in page.get_images(full=True):
                try:
                    pix = fitz.Pixmap(doc, item[0])
                    if pix.n - pix.alpha >= 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    images.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
                except Exception:
                    pass
            if len(images) >= MAX_IMAGES:
                break
        if not [im for im in images if min(im.size) >= MIN_SIDE]:
            images = []
            for page in list(doc)[:3]:          # scanned/vector PDF: render pages
                pix = page.get_pixmap(dpi=150)
                images.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))

    elif name.endswith(".docx"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for n in z.namelist():
                if n.startswith("word/media/"):
                    try:
                        images.append(Image.open(io.BytesIO(z.read(n))).convert("RGB"))
                    except Exception:
                        pass                      # skip unsupported formats (emf/wmf)

    elif name.endswith(".doc"):
        raise ValueError("Old .doc files are not supported. Open it in Word, "
                         "choose Save As > .docx (or PDF), and upload that file.")
    else:
        images.append(Image.open(io.BytesIO(data)).convert("RGB"))

    return [im for im in images if min(im.size) >= MIN_SIDE][:MAX_IMAGES]


# ---------------------------------------------------------------
# Approximate tumor size (simple image-processing estimate)
# ---------------------------------------------------------------
def _otsu(values):
    if values.size < 50:
        return float(values.mean()) if values.size else 0.0
    hist, edges = np.histogram(values, bins=128)
    hist = hist.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2
    w1 = np.cumsum(hist)
    w2 = w1[-1] - w1
    s = np.cumsum(hist * centers)
    m1 = s / np.maximum(w1, 1e-9)
    m2 = (s[-1] - s) / np.maximum(w2, 1e-9)
    var = w1 * w2 * (m1 - m2) ** 2
    return float(centers[int(np.argmax(var[:-1]))])


def estimate_tumor(img, fov_mm):
    """Rough tumor size from the brightest connected region inside the brain.
    Returns None if nothing reliable is found."""
    from scipy import ndimage as ndi

    W, H = img.size
    N = 256
    gray = np.asarray(img.convert("L").resize((N, N)), dtype=float)
    gray = ndi.gaussian_filter(gray, 1.0)
    g = gray / max(gray.max(), 1.0)

    head = ndi.binary_fill_holes(ndi.binary_opening(g > 0.10, iterations=2))
    lab, n = ndi.label(head)
    if n == 0:
        return None
    sizes = ndi.sum(head, lab, range(1, n + 1))
    head = lab == (int(np.argmax(sizes)) + 1)
    brain = ndi.binary_erosion(head, iterations=12)      # drop skull edge
    if brain.sum() < 2000:
        return None

    vals = g[brain]
    t1 = _otsu(vals)
    t2 = _otsu(vals[vals > t1])
    cand = ndi.binary_opening(brain & (g > t2), iterations=2)
    lab, n = ndi.label(cand)
    if n == 0:
        return None
    sizes = ndi.sum(cand, lab, range(1, n + 1))
    idx = int(np.argmax(sizes)) + 1
    comp = lab == idx
    area_px = float(comp.sum())
    if area_px < 0.002 * brain.sum():
        return None

    mm_x = fov_mm / N                      # image width spans the field of view
    mm_y = fov_mm * (H / W) / N            # assume square pixels
    area_mm2 = area_px * mm_x * mm_y
    diameter = 2.0 * np.sqrt(area_mm2 / np.pi)
    sl = ndi.find_objects(comp.astype(int))[0]
    ext_y = (sl[0].stop - sl[0].start) * mm_y
    ext_x = (sl[1].stop - sl[1].start) * mm_x

    # length = longest axis, width = perpendicular axis (ellipse fitted to the region)
    ys, xs = np.nonzero(comp)
    pts = np.stack([xs * mm_x, ys * mm_y], axis=1)
    ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1]
    length = 4.0 * np.sqrt(max(ev[0], 0.0))
    width = 4.0 * np.sqrt(max(ev[1], 0.0))

    base = np.asarray(img.resize((N, N)).convert("RGB")).copy()
    edge = comp & ~ndi.binary_erosion(comp, iterations=2)
    base[edge] = [255, 60, 60]
    return {
        "length_mm": float(length),
        "width_mm": float(width),
        "diameter_mm": float(diameter),
        "extent_mm": (float(ext_x), float(ext_y)),
        "area_mm2": float(area_mm2),
        "pct_brain": 100.0 * area_px / float(brain.sum()),
        "overlay": Image.fromarray(base).resize((384, 384)),
    }


PAGE_DASH = "🏠 Dashboard"
PAGE_SCAN = "🧠 MRI Scan"
PAGE_STATS = "📊 Usage Statistics"


# ---------------------------------------------------------------
# Full-screen glittering welcome (shown once right after sign in)
# ---------------------------------------------------------------
def welcome_screen(name):
    rnd = random.Random(7)
    sparks = []
    for _ in range(110):
        s = rnd.uniform(2, 7)
        sparks.append(
            f'<span class="sp" style="left:{rnd.uniform(0, 100):.1f}%;top:{rnd.uniform(0, 100):.1f}%;'
            f'width:{s:.1f}px;height:{s:.1f}px;animation-delay:{rnd.uniform(0, 3):.2f}s;'
            f'animation-duration:{rnd.uniform(1.2, 3):.2f}s"></span>'
        )
    safe_name = name.replace("<", "").replace(">", "")
    st.markdown(
        f"""
        <style>
        .welcome-overlay {{
            position: fixed; inset: 0; z-index: 99999999;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            background: radial-gradient(circle at 50% 40%, #14407f 0%, #07152b 55%, #030a16 100%);
            overflow: hidden; animation: fadein .6s ease-out;
        }}
        @keyframes fadein {{ from {{ opacity: 0 }} to {{ opacity: 1 }} }}
        .sp {{
            position: absolute; border-radius: 50%; background: #fff; opacity: 0;
            box-shadow: 0 0 10px 3px rgba(255,215,120,.9);
            animation-name: twinkle; animation-iteration-count: infinite; animation-timing-function: ease-in-out;
        }}
        @keyframes twinkle {{ 0%,100% {{ opacity: 0; transform: scale(.3) }} 50% {{ opacity: 1; transform: scale(1.5) }} }}
        .w-brain {{ font-size: 4.5rem; animation: pulse 1.6s ease-in-out infinite; z-index: 2; }}
        @keyframes pulse {{ 0%,100% {{ transform: scale(1) }} 50% {{ transform: scale(1.15) }} }}
        .w-title {{
            z-index: 2; font-size: clamp(2.4rem, 8vw, 5.5rem); font-weight: 800; letter-spacing: .04em;
            background: linear-gradient(90deg, #ffe29a, #ffffff, #ffd36b, #ffffff, #ffe29a);
            background-size: 200% auto; -webkit-background-clip: text; background-clip: text;
            -webkit-text-fill-color: transparent; animation: shine 2.4s linear infinite;
        }}
        @keyframes shine {{ to {{ background-position: 200% center }} }}
        .w-name {{ z-index: 2; font-size: clamp(1.4rem, 4vw, 2.4rem); color: #bfe9ff; margin-top: .3rem; }}
        .w-sub {{ z-index: 2; margin-top: 1.2rem; color: #8fd8ff; letter-spacing: .25em; font-size: .95rem; text-transform: uppercase; }}
        </style>
        <div class="welcome-overlay">
            {''.join(sparks)}
            <div class="w-brain">🧠</div>
            <div class="w-title">Welcome</div>
            <div class="w-name">{safe_name}</div>
            <div class="w-sub">MRI NeuroScan AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(3.5)
    st.rerun()


# ---------------------------------------------------------------
# Classifier page (shown after login)
# ---------------------------------------------------------------
def usage_statistics():
    """Usage statistics for the dashboard, protected by ADMIN_PASSWORD (Streamlit secrets)."""
    admin_pw = _secret("ADMIN_PASSWORD")
    if not admin_pw:
        return
    st.markdown("### 📊 Usage statistics")
    if not st.session_state.get("admin_ok"):
        entered = st.text_input("Admin password", type="password", key="admin_pw")
        if not entered:
            return
        if not hmac.compare_digest(entered, str(admin_pw)):
            st.error("Wrong admin password.")
            return
        st.session_state["admin_ok"] = True
    try:
        s = store_stats()
        rows = store_recent(100)
    except Exception:
        st.error("Could not read statistics from the database.")
        return
    if not USE_SUPABASE:
        st.warning("Temporary storage: these numbers reset when the app restarts.")
    c1, c2 = st.columns(2)
    c1.metric("Registered users", s["users"])
    c2.metric("Total logins", s["logins"])
    c3, c4 = st.columns(2)
    c3.metric("Unique users who logged in", s["unique_logins"])
    c4.metric("Scans analysed", s["analyses"])
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download log (CSV)", df.to_csv(index=False).encode(),
                           "usage_log.csv", "text/csv")


def goto(page):
    st.session_state["page"] = page


def dashboard_page():
    st.markdown(
        """
        <style>
        .nav-card { background: rgba(255,255,255,.07); border: 1px solid rgba(95,212,255,.35);
                    border-radius: 16px; padding: 1.4rem 1rem 1rem 1rem; text-align: center;
                    min-height: 190px; }
        .nav-card .ico { font-size: 2.8rem; }
        .nav-card h3 { margin: .3rem 0 .4rem 0; }
        .nav-card p { font-size: .9rem; opacity: .85; margin: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 🏠 Main Dashboard")
    st.write("Choose what you would like to open:")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="nav-card"><div class="ico">🧠</div><h3>MRI Scan</h3>'
            '<p>Upload an MRI image, PDF or Word file and get the tumor classification, '
            'approximate size and a report.</p></div>',
            unsafe_allow_html=True,
        )
        st.button("Open MRI Scan", type="primary", use_container_width=True,
                  on_click=goto, args=(PAGE_SCAN,), key="open_scan")
    with c2:
        st.markdown(
            '<div class="nav-card"><div class="ico">📊</div><h3>Usage Statistics</h3>'
            '<p>See how many people registered, logged in and analysed scans '
            '(admin password required).</p></div>',
            unsafe_allow_html=True,
        )
        st.button("Open Usage Statistics", type="primary", use_container_width=True,
                  on_click=goto, args=(PAGE_STATS,), key="open_stats")


def stats_page():
    if not _secret("ADMIN_PASSWORD"):
        st.info("Usage statistics are not enabled yet. Add ADMIN_PASSWORD in the app's "
                "Streamlit Secrets to turn them on.")
        return
    usage_statistics()


def app_shell():
    """Header and page routing shown after login."""
    _, top_r = st.columns([4, 1])
    with top_r:
        if st.button("Log out", use_container_width=True):
            for k_ in ("user", "username", "pred", "page", "admin_ok"):
                st.session_state.pop(k_, None)
            st.rerun()
    st.markdown('<div class="brand"><h1>🧠 MRI NeuroScan AI</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Brain tumor MRI classification · CNN + ZOA + SVM</div>',
                unsafe_allow_html=True)

    page = st.session_state.get("page", PAGE_DASH)
    if page == PAGE_DASH:
        dashboard_page()
        return
    st.button("← Back to Dashboard", on_click=goto, args=(PAGE_DASH,), key="back_home")
    if page == PAGE_SCAN:
        classifier_page()
    else:
        stats_page()


def classifier_page():
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    bundle, cnn = load_models()
    scaler, mask, svm, classes = (bundle["scaler"], bundle["mask"],
                                  bundle["svm"], bundle["classes"])

    uploaded = st.file_uploader(
        "Upload MRI (JPG, PNG, BMP, TIFF, WEBP, PDF, DOCX)", type=UPLOAD_TYPES)
    if uploaded is None:
        st.warning("For research and educational use only. Not a substitute for professional medical diagnosis.")
        return

    try:
        images = images_from_file(uploaded)
    except Exception as e:
        st.error(f"Could not read this file: {e}")
        return
    if not images:
        st.error("No usable image was found in this file.")
        return

    if len(images) > 1:
        i = st.selectbox(f"{len(images)} images found - choose one", range(len(images)),
                         format_func=lambda x: f"Image {x + 1}")
    else:
        i = 0
    img = images[i]
    key = (uploaded.name, uploaded.size, i)

    # Analyze button is at the top, right under the uploader
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            arr = preprocess_input(
                np.array(img.resize((224, 224)), dtype="float32")[None, ...])
            feats = cnn.predict(arr, verbose=0)       # CNN features (1280)
            X = scaler.transform(feats)[:, mask]      # ZOA-selected features
            proba = svm.predict_proba(X)[0]           # SVM classification
        kk = int(proba.argmax())
        st.session_state["pred"] = {"key": key, "k": kk, "proba": proba}
        store_log(st.session_state.get("username", ""), "analysis",
                  f"{LABELS.get(classes[kk], classes[kk])} {proba[kk] * 100:.1f}%")

    fov = FOV_MM

    pred = st.session_state.get("pred")
    if pred and pred["key"] != key:
        pred = None

    tumor = None
    if pred and classes[pred["k"]] != "notumor":
        tumor = estimate_tumor(img, fov)

    col_img, col_rep = st.columns(2)

    with col_img:
        st.image(img, caption="Selected MRI image", use_container_width=True)
        if tumor is not None:
            st.image(tumor["overlay"], caption="Region used for size estimate (red outline)",
                     use_container_width=True)

    with col_rep:
        st.markdown("### 📋 Analysis Report")
        if pred is None:
            st.info("Click **Analyze** to generate the report for this image.")
        else:
            k, proba = pred["k"], pred["proba"]
            ist = timezone(timedelta(hours=5, minutes=30))
            st.caption(f"File: {uploaded.name}  |  Generated: "
                       f"{datetime.now(ist).strftime('%d %b %Y, %I:%M %p')} IST")
            st.subheader(LABELS.get(classes[k], classes[k]))
            st.write(f"Confidence: **{proba[k] * 100:.2f}%**")
            for c, p in zip(classes, proba):
                st.write(f"{LABELS.get(c, c)} - {p * 100:.1f}%")
                st.progress(float(p))
            st.caption(f"ZOA selected {int(mask.sum())} of {len(mask)} CNN features.")

            if classes[k] != "notumor":
                st.markdown("#### 📏 Tumor size (approximate)")
                if tumor is None:
                    st.warning("Tumor detected, but its size could not be estimated "
                               "from this image.")
                else:
                    m1, m2 = st.columns(2)
                    m1.metric("Length", f"{tumor['length_mm']:.1f} mm",
                              f"{tumor['length_mm'] / 10:.2f} cm", delta_color="off")
                    m2.metric("Width", f"{tumor['width_mm']:.1f} mm",
                              f"{tumor['width_mm'] / 10:.2f} cm", delta_color="off")
                    st.caption("Length = longest dimension, width = perpendicular dimension. "
                               "Estimated from a single 2D image with an assumed scale, "
                               "so it is approximate, not a clinical measurement.")
                st.info("🩺 **Important:** This is an automated screening result, not a diagnosis. "
                        "Please share this report with a doctor (neurologist, neurosurgeon or "
                        "radiologist). The doctor will confirm the findings on the original scans "
                        "and decide the required tests, treatment and precautions.")
            else:
                st.success("✅ No tumor was detected in this image. This does not replace a medical "
                           "opinion - if you have symptoms such as persistent headache, seizures or "
                           "vision problems, please consult a doctor.")

    st.warning("For research and educational use only. Not a substitute for professional medical diagnosis.")


# ---------------------------------------------------------------
set_background()
if "user" in st.session_state:
    if st.session_state.pop("show_welcome", False):
        welcome_screen(st.session_state["user"])   # shows ~3.5 s, then reruns
    else:
        app_shell()
else:
    auth_page()
