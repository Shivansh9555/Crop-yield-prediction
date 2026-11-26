"""
AgriYield AI v2 - Smart Crop Recommendation using Weather + Soil Data
----------------------------------------------------------------------
* Predicts the BEST CROP to grow from: soil (N, P, K, pH) + weather (temperature, humidity, rainfall)
* Model trains AUTOMATICALLY on first run (no manual step) and is saved to disk (models/crop_model.joblib)
* Auto re-trains whenever the dataset changes (e.g., you upload your own CSV)
* Trains 5 algorithms, compares Accuracy / Precision / Recall / F1, and picks the best one automatically
* Optional: fetches LIVE weather (temperature, humidity, last-30-day rainfall) from the free Open-Meteo API

Run:
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib requests
    streamlit run main.py
"""

import os
import hashlib

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ----------------------------------------------------------------------------
# 0. CONSTANTS
# ----------------------------------------------------------------------------
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.joblib")

st.set_page_config(page_title="AgriYield AI - Crop Recommendation", page_icon="🌾", layout="wide",
                   initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------
# 0b. DESIGN - custom CSS theme + small UI helpers
# ----------------------------------------------------------------------------
CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘", "pigeonpeas": "🫛",
    "mothbeans": "🫘", "mungbean": "🫛", "blackgram": "🫘", "lentil": "🫘", "pomegranate": "🍎",
    "banana": "🍌", "mango": "🥭", "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈",
    "apple": "🍏", "orange": "🍊", "papaya": "🥭", "coconut": "🥥", "cotton": "☁️",
    "jute": "🌿", "coffee": "☕",
}

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1250px;}
    #MainMenu, footer {visibility: hidden;}

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 45%, #66bb6a 100%);
        border-radius: 22px; padding: 2.2rem 2.4rem; color: #fff; margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px rgba(27,94,32,.35); position: relative; overflow: hidden;
    }
    .hero::after {
        content: "🌱"; position: absolute; right: 28px; bottom: -18px; font-size: 130px; opacity: .18;
    }
    .hero h1 {margin: 0; font-size: 2.3rem; font-weight: 800; color: #fff;}
    .hero p {margin: .5rem 0 0; font-size: 1.05rem; opacity: .95; color: #fff;}
    .badge {
        display: inline-block; background: rgba(255,255,255,.2); border: 1px solid rgba(255,255,255,.35);
        border-radius: 999px; padding: .2rem .8rem; margin: .8rem .4rem 0 0; font-size: .82rem; color: #fff;
    }

    /* KPI cards */
    .kpi {
        background: rgba(46,125,50,.08); border: 1px solid rgba(46,125,50,.25);
        border-left: 6px solid #2e7d32; border-radius: 14px; padding: .9rem 1.1rem;
    }
    .kpi .v {font-size: 1.6rem; font-weight: 800; color: #2e7d32; line-height: 1.2;}
    .kpi .l {font-size: .82rem; opacity: .75; text-transform: uppercase; letter-spacing: .05em;}

    /* Section titles */
    .sec {
        display: flex; align-items: center; gap: .6rem; font-size: 1.25rem; font-weight: 700;
        margin: 1.2rem 0 .6rem; padding-bottom: .35rem; border-bottom: 2px solid rgba(46,125,50,.35);
    }
    .sec span.dot {
        background: #2e7d32; color: #fff; width: 1.9rem; height: 1.9rem; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center; font-size: .95rem;
    }

    /* Result card */
    .result {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9); color: #1b5e20; border-radius: 20px;
        padding: 1.6rem 2rem; text-align: center; border: 2px solid #66bb6a;
        box-shadow: 0 8px 24px rgba(46,125,50,.25); margin: 1rem 0;
    }
    .result .emoji {font-size: 4rem; line-height: 1;}
    .result .name {font-size: 2.2rem; font-weight: 800; margin: .3rem 0 0;}
    .result .sub {font-size: 1rem; opacity: .85;}
    .meter {background: rgba(27,94,32,.15); border-radius: 999px; height: 12px; margin: .8rem auto 0; max-width: 360px;}
    .meter > div {background: linear-gradient(90deg,#43a047,#1b5e20); height: 12px; border-radius: 999px;}

    /* Buttons & tabs */
    div.stButton > button {
        border-radius: 12px; font-weight: 700; padding: .55rem 1.4rem; border: 1px solid #2e7d32;
        transition: all .15s ease;
    }
    div.stButton > button:hover {transform: translateY(-2px); box-shadow: 0 6px 14px rgba(46,125,50,.35);}
    div.stButton > button[kind="primary"] {background: linear-gradient(90deg,#2e7d32,#43a047); color: #fff; border: none;}
    button[data-baseweb="tab"] {font-weight: 700; font-size: 1rem;}
    [data-testid="stSidebar"] {border-right: 2px solid rgba(46,125,50,.25);}

    /* Soft background glow */
    [data-testid="stAppViewContainer"] {
        background-image: radial-gradient(circle at 12% 8%, rgba(102,187,106,.14), transparent 38%),
                          radial-gradient(circle at 92% 90%, rgba(46,125,50,.12), transparent 40%);
    }

    /* Input panels (st.container(border=True)) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px !important; border: 1px solid rgba(46,125,50,.30) !important;
        background: rgba(46,125,50,.04); box-shadow: 0 4px 14px rgba(46,125,50,.08);
    }

    /* Sidebar brand */
    .brand {
        display: flex; align-items: center; gap: .7rem; padding: .8rem 1rem; border-radius: 14px;
        background: linear-gradient(135deg,#2e7d32,#66bb6a); color: #fff; margin-bottom: 1rem;
    }
    .brand .logo {font-size: 2rem;}
    .brand .t {font-weight: 800; font-size: 1.1rem; line-height: 1.1;}
    .brand .s {font-size: .75rem; opacity: .9;}

    /* Footer */
    .footer {
        text-align: center; margin-top: 2.5rem; padding: 1rem; border-radius: 14px;
        background: rgba(46,125,50,.08); font-size: .88rem; opacity: .9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
    .stMarkdown, .navbar, .hero2, .feat, .step, .title-block, .footer2, .result, .kpi, .sec,
    .stTabs button, div.stButton button, label p {font-family: 'Poppins', sans-serif;}
    [data-testid="stHeader"] {background: transparent;}

    /* Navbar */
    .navbar {display:flex; justify-content:space-between; align-items:center; padding:.8rem 1.3rem;
        border-radius:16px; margin-bottom:1rem; background:rgba(46,125,50,.10);
        border:1px solid rgba(46,125,50,.25); backdrop-filter: blur(8px);}
    .navbar .brand-n {font-weight:800; font-size:1.25rem; color:#2e7d32;}
    .navbar .right {display:flex; gap:.5rem; flex-wrap:wrap;}
    .navbar .pill {background:#2e7d32; color:#fff; padding:.25rem .85rem; border-radius:999px;
        font-size:.8rem; font-weight:600;}
    .navbar .pill.ghost {background:transparent; color:#2e7d32; border:1px solid #2e7d32;}

    /* Hero */
    .hero2 {display:grid; grid-template-columns:1.4fr 1fr; gap:1rem; align-items:center;
        border-radius:28px; padding:3rem; color:#fff; margin-bottom:1.5rem; position:relative; overflow:hidden;
        background:linear-gradient(120deg,#0b3d12 0%,#1b5e20 35%,#43a047 100%);
        box-shadow:0 20px 50px rgba(27,94,32,.4);}
    .hero2 h1 {font-size:3rem; font-weight:800; line-height:1.1; margin:0 0 .8rem; color:#fff;}
    .hero2 h1 span {color:#c5e1a5;}
    .hero2 p {font-size:1.1rem; opacity:.92; color:#fff; margin:0 0 1.3rem;}
    .cta {display:inline-block; background:#fff; color:#1b5e20; font-weight:700; padding:.7rem 1.6rem;
        border-radius:999px; box-shadow:0 8px 20px rgba(0,0,0,.25);}
    .float-wrap {position:relative; height:260px;}
    .float-wrap .f {position:absolute; font-size:3.4rem; animation: floaty 5s ease-in-out infinite;
        filter: drop-shadow(0 8px 10px rgba(0,0,0,.3));}
    @keyframes floaty {0%,100% {transform:translateY(0) rotate(0deg);} 50% {transform:translateY(-18px) rotate(6deg);}}

    /* Feature cards */
    .feat-grid {display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:.5rem 0 1.5rem;}
    .feat {background:rgba(46,125,50,.07); border:1px solid rgba(46,125,50,.22); border-radius:18px;
        padding:1.3rem; transition:all .2s ease;}
    .feat:hover {transform:translateY(-6px); box-shadow:0 12px 26px rgba(46,125,50,.25);}
    .feat .ic {font-size:2.2rem;}
    .feat h4 {margin:.5rem 0 .3rem; color:#2e7d32; font-size:1.05rem;}
    .feat p {margin:0; font-size:.88rem; opacity:.8;}

    /* How it works */
    .steps {display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1rem;}
    .step {display:flex; gap:1rem; align-items:flex-start; padding:1.1rem; border-radius:18px;
        background:linear-gradient(135deg,rgba(46,125,50,.10),rgba(102,187,106,.06));
        border:1px dashed rgba(46,125,50,.4);}
    .step .n {flex:0 0 2.6rem; height:2.6rem; border-radius:50%; background:#2e7d32; color:#fff;
        font-weight:800; display:flex; align-items:center; justify-content:center; font-size:1.2rem;}
    .step h4 {margin:0 0 .2rem; font-size:1rem;}
    .step p {margin:0; font-size:.86rem; opacity:.8;}

    /* Centered titles */
    .title-block {text-align:center; margin:2rem 0 1rem;}
    .title-block h2 {font-size:2rem; font-weight:800; margin:0;}
    .title-block p {opacity:.75; margin:.3rem 0 0;}
    .title-block .bar {width:70px; height:5px; border-radius:5px; margin:.7rem auto 0;
        background:linear-gradient(90deg,#2e7d32,#a5d6a7);}

    /* Footer */
    .footer2 {margin-top:3rem; border-radius:24px; padding:2rem; color:#fff;
        background:linear-gradient(120deg,#0b3d12,#1b5e20); display:grid;
        grid-template-columns:1.5fr 1fr 1fr; gap:1.5rem;}
    .footer2 h4 {margin:0 0 .6rem; color:#c5e1a5;}
    .footer2 p, .footer2 li {font-size:.88rem; opacity:.92; color:#fff; margin:.2rem 0;}
    .footer2 ul {list-style:none; padding:0; margin:0;}
    .copy {text-align:center; opacity:.7; font-size:.8rem; margin-top:.8rem;}

    @media (max-width: 900px) {
        .hero2 {grid-template-columns:1fr; padding:2rem;} .hero2 h1 {font-size:2.1rem;}
        .feat-grid {grid-template-columns:1fr 1fr;} .steps {grid-template-columns:1fr;}
        .footer2 {grid-template-columns:1fr;} .float-wrap {display:none;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def section(num: str, title: str):
    st.markdown(f'<div class="sec"><span class="dot">{num}</span>{title}</div>', unsafe_allow_html=True)


def kpi(col, value: str, label: str):
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 1. DATA  (built-in synthetic dataset OR user-uploaded CSV)
# ----------------------------------------------------------------------------
# Typical growing conditions per crop: N, P, K, temp(C), humidity(%), pH, rainfall(mm)
CROP_PROFILES = {
    "rice":        (80, 48, 40, 23.7, 82, 6.4, 236),
    "maize":       (77, 48, 20, 22.4, 65, 6.2, 85),
    "chickpea":    (40, 68, 80, 18.9, 17, 7.3, 80),
    "kidneybeans": (21, 68, 20, 20.0, 21, 5.8, 106),
    "pigeonpeas":  (21, 68, 20, 27.7, 49, 5.8, 149),
    "mothbeans":   (21, 48, 20, 28.2, 53, 6.8, 51),
    "mungbean":    (21, 48, 20, 28.5, 85, 6.7, 48),
    "blackgram":   (40, 67, 19, 30.0, 65, 7.1, 68),
    "lentil":      (19, 68, 19, 24.5, 64, 6.9, 46),
    "pomegranate": (19, 19, 40, 21.8, 90, 6.4, 107),
    "banana":      (100, 82, 50, 27.4, 80, 6.0, 105),
    "mango":       (21, 21, 30, 31.2, 50, 5.8, 95),
    "grapes":      (23, 133, 200, 23.8, 82, 6.0, 70),
    "watermelon":  (99, 17, 50, 25.6, 85, 6.5, 50),
    "muskmelon":   (100, 17, 50, 28.7, 92, 6.4, 25),
    "apple":       (21, 134, 200, 22.6, 92, 6.0, 113),
    "orange":      (20, 10, 10, 22.8, 92, 7.0, 110),
    "papaya":      (50, 59, 50, 33.7, 92, 6.7, 143),
    "coconut":     (22, 17, 31, 27.4, 95, 6.0, 175),
    "cotton":      (118, 46, 20, 24.0, 80, 6.9, 80),
    "jute":        (78, 46, 40, 25.0, 80, 6.7, 175),
    "coffee":      (101, 29, 30, 25.5, 58, 6.8, 158),
}


@st.cache_data
def generate_builtin_dataset(samples_per_crop: int = 100) -> pd.DataFrame:
    """Creates a realistic synthetic dataset around each crop's ideal conditions."""
    rng = np.random.default_rng(42)
    rows = []
    for crop, (n, p, k, t, h, ph, r) in CROP_PROFILES.items():
        means = np.array([n, p, k, t, h, ph, r], dtype=float)
        stds = np.array([
            max(5, 0.12 * n), max(4, 0.12 * p), max(4, 0.10 * k),
            2.0, 4.0, 0.4, max(8, 0.15 * r),
        ])
        data = rng.normal(means, stds, size=(samples_per_crop, 7))
        for row in data:
            rows.append([*row, crop])
    df = pd.DataFrame(rows, columns=FEATURES + [TARGET])
    df["N"] = df["N"].clip(0, 140)
    df["P"] = df["P"].clip(5, 145)
    df["K"] = df["K"].clip(5, 205)
    df["temperature"] = df["temperature"].clip(8, 44)
    df["humidity"] = df["humidity"].clip(10, 100)
    df["ph"] = df["ph"].clip(3.5, 9.9)
    df["rainfall"] = df["rainfall"].clip(20, 300)
    return df.sample(frac=1, random_state=42).reset_index(drop=True).round(2)


def normalise_uploaded_csv(raw: pd.DataFrame) -> pd.DataFrame:
    """Accepts the popular Kaggle 'Crop_recommendation.csv' layout (case-insensitive)."""
    mapping = {
        "n": "N", "nitrogen": "N", "p": "P", "phosphorus": "P", "k": "K", "potassium": "K",
        "temperature": "temperature", "temp": "temperature",
        "humidity": "humidity", "ph": "ph", "rainfall": "rainfall",
        "label": TARGET, "crop": TARGET,
    }
    raw = raw.rename(columns={c: mapping.get(c.strip().lower(), c) for c in raw.columns})
    missing = [c for c in FEATURES + [TARGET] if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}. Required: {FEATURES + [TARGET]}")
    return raw[FEATURES + [TARGET]].dropna().reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. AUTOMATIC TRAINING
# ----------------------------------------------------------------------------
def build_models() -> dict:
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "SVM (RBF)": Pipeline([("scale", StandardScaler()), ("clf", SVC(probability=True, random_state=42))]),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }


def dataset_signature(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def train_all(df: pd.DataFrame, signature: str) -> dict:
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    fitted, rows = {}, []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="weighted", zero_division=0
        )
        rows.append({
            "Algorithm": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
        })
        fitted[name] = model

    metrics = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    best_name = metrics.loc[0, "Algorithm"]

    classes = sorted(y.unique())
    cm = confusion_matrix(y_test, fitted[best_name].predict(X_test), labels=classes)

    bundle = {
        "signature": signature,
        "models": fitted,
        "metrics": metrics,
        "best_name": best_name,
        "classes": classes,
        "confusion_matrix": cm,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_ranges": {f: (float(df[f].min()), float(df[f].max())) for f in FEATURES},
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


@st.cache_resource(show_spinner="Training models automatically... (only happens once per dataset)")
def get_bundle(signature: str, _df: pd.DataFrame, retrain_token: int) -> dict:
    """Loads the saved model if it matches the dataset, otherwise trains automatically."""
    if retrain_token == 0 and os.path.exists(MODEL_PATH):
        try:
            saved = joblib.load(MODEL_PATH)
            if saved.get("signature") == signature:
                return saved
        except Exception:
            pass  # corrupted / incompatible file -> retrain
    return train_all(_df, signature)


# ----------------------------------------------------------------------------
# 3. EXTRA FEATURES - assistant, regional scan, printable report
# ----------------------------------------------------------------------------
FEATURE_LABELS = {
    "N": "Nitrogen N (kg/ha)", "P": "Phosphorus P (kg/ha)", "K": "Potassium K (kg/ha)",
    "temperature": "Temperature (°C)", "humidity": "Humidity (%)", "ph": "Soil pH", "rainfall": "Rainfall (mm)",
}

# General sowing-season guidance (varies by region and variety)
CROP_SEASON = {
    "rice": "Kharif (Jun-Oct)", "maize": "Kharif and Rabi", "chickpea": "Rabi (Oct-Mar)",
    "kidneybeans": "Kharif (Jun-Sep)", "pigeonpeas": "Kharif (Jun-Dec)", "mothbeans": "Kharif (Jul-Oct)",
    "mungbean": "Kharif and Zaid", "blackgram": "Kharif and Rabi", "lentil": "Rabi (Oct-Mar)",
    "pomegranate": "Perennial orchard", "banana": "Perennial (plant Feb-Mar or Jun-Jul)",
    "mango": "Perennial orchard", "grapes": "Perennial vineyard", "watermelon": "Zaid (Feb-Jun)",
    "muskmelon": "Zaid (Feb-Jun)", "apple": "Perennial orchard (temperate regions)",
    "orange": "Perennial orchard", "papaya": "Perennial (plant Feb-Mar or Sep-Oct)",
    "coconut": "Perennial plantation", "cotton": "Kharif (May-Dec)", "jute": "Kharif (Mar-Jul)",
    "coffee": "Perennial plantation (hilly regions)",
}


def predict_yield_one(crop: str, inputs: dict) -> float:
    """Expected yield (t/ha) for one crop using the auto-trained yield model."""
    row = pd.DataFrame([{"crop": crop, **{f: inputs[f] for f in FEATURES}}])[["crop"] + FEATURES]
    return float(max(0.0, yield_bundle["model"].predict(row)[0]))


def assistant_reply(q: str, last) -> str:
    """Small offline rule-based farming assistant that uses the latest prediction as context."""
    t = q.lower().strip()
    has = lambda *words: any(w in t for w in words)
    crop = last["crop"] if last else None
    need_pred = "Run a prediction in the 🌱 Crop Predictor tab first, then I can give you a personalised answer."

    if t in ("hi", "hey", "hello", "namaste") or t.startswith(("hello", "namaste")):
        return "Namaste! 🌾 Ask me about fertilizer, water, season, soil pH, yield, profit or pests."
    if has("why") and crop:
        return (f"**{crop.title()}** was recommended because your conditions - {last['temperature']} °C, "
                f"{last['humidity']}% humidity, {last['rainfall']} mm rainfall, N-P-K "
                f"{last['N']}-{last['P']}-{last['K']} and pH {last['ph']} - are closest to what "
                f"{crop.title()} typically needs. Use the 🔬 What-If tab to see which input matters most.")
    if has("fertili", "urea", "dap", "npk", "nutrient", "nitrogen", "phosph", "potash", "lime", "acid", "alkaline", " ph"):
        if crop not in CROP_PROFILES:
            return need_pred
        items, score = soil_advice(crop, last["N"], last["P"], last["K"], last["ph"])
        lines = [f"- **{label}**: {status} - {tip}" for label, _, _, status, _, tip in items if status != "Adequate"]
        return (f"Soil health score for **{crop.title()}**: {score:.0f}/100.\n" +
                ("\n".join(lines) if lines else "Your N, P, K and pH are all in a suitable range 👍"))
    if has("water", "irrigat", "rain"):
        if crop not in CROP_PROFILES:
            return need_pred
        r = CROP_PROFILES[crop][6]
        level = "high" if r >= 150 else "moderate" if r >= 80 else "low"
        return (f"**{crop.title()}** typically needs {level} water (about {r} mm on this dataset's scale). "
                f"Your rainfall input is {last['rainfall']} mm. Irrigate to cover any gap and avoid waterlogging.")
    if has("season", "sow", "when", "plant"):
        if crop not in CROP_SEASON:
            return need_pred
        return f"**{crop.title()}**: {CROP_SEASON[crop]}. General guidance - confirm exact dates with your local agriculture office."
    if has("profit", "price", "earn", "income", "cost", "money"):
        if crop not in CROP_ECONOMICS:
            return need_pred
        y = predict_yield_one(crop, last)
        _, price, cost = CROP_ECONOMICS[crop]
        profit = y * 1000 * price - cost
        sign = "-" if profit < 0 else ""
        return (f"Rough estimate for **{crop.title()}** on 1 hectare: yield ~{y:.2f} t/ha, profit about "
                f"{sign}₹{abs(profit):,.0f} using illustrative price and cost. Edit them in the 💰 Yield & Profit tab.")
    if has("yield", "production", "harvest"):
        if crop not in CROP_ECONOMICS:
            return need_pred
        return f"Expected yield for **{crop.title()}** with your inputs is about **{predict_yield_one(crop, last):.2f} tons/hectare**."
    if has("pest", "disease", "insect", "fungus", "leaf", "spot"):
        return ("Use integrated pest management: scout fields weekly, rotate crops, use resistant varieties, "
                "try neem-based sprays first and keep fields clean. For a specific disease, show a sample to your "
                "local Krishi Vigyan Kendra (KVK) or agriculture officer before spraying anything.")
    if has("help", "what can"):
        return "I can answer about: why this crop, fertilizer and pH, water, season, yield, profit and pests."
    return ("I'm a simple offline assistant. Try: *why this crop?*, *which fertilizer?*, *how much water?*, "
            "*what is the season?*, *expected profit?* or *pest control?*")


def build_report_html(last: dict, items: list, econ) -> str:
    """Printable HTML report (open in browser -> Print -> Save as PDF)."""
    inputs = "".join(f"<tr><td>{FEATURE_LABELS[f]}</td><td>{last[f]}</td></tr>" for f in FEATURES)
    top3 = "".join(f"<li>{c.title()} - {p:.1%}</li>" for c, p in last.get("top3_list", []))
    adv = "".join(f"<tr><td>{a[0]}</td><td>{a[3]}</td><td>{a[5]}</td></tr>" for a in items)
    econ_html = ""
    if econ:
        econ_html = (f"<h2>Expected yield &amp; profit (illustrative)</h2><p>Yield: <b>{econ['yield']:.2f} t/ha</b> | "
                     f"Revenue: Rs {econ['revenue']:,.0f} | Cost: Rs {econ['cost']:,.0f} | "
                     f"Profit: <b>Rs {econ['profit']:,.0f}</b> per hectare</p>")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Crop Recommendation Report</title>
<style>body{{font-family:Arial,sans-serif;max-width:800px;margin:30px auto;color:#222}}
h1{{color:#2e7d32}} h2{{color:#2e7d32;border-bottom:2px solid #a5d6a7;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #c8e6c9;padding:6px 10px;text-align:left}}
.box{{background:#e8f5e9;border:2px solid #66bb6a;border-radius:12px;padding:16px;text-align:center;font-size:22px}}</style></head>
<body><h1>🌾 AgriYield AI - Crop Recommendation Report</h1><p>Generated: {last['time']}</p>
<div class="box">Recommended crop: <b>{last['crop'].title()}</b> ({last['confidence']:.1%} confidence, model: {last['model']})</div>
<h2>Input data</h2><table>{inputs}</table>
<h2>Top suggestions</h2><ul>{top3}</ul>
<h2>Soil &amp; fertilizer advice</h2><table><tr><th>Parameter</th><th>Status</th><th>Advice</th></tr>{adv}</table>
{econ_html}
<p style="font-size:12px;color:#777">General guidance from a machine learning model trained on synthetic data. Confirm with a local soil test and your agriculture officer.</p>
</body></html>"""


def report_for_last(last: dict) -> str:
    items, econ = [], None
    crop = last["crop"]
    if crop in CROP_PROFILES:
        items, _ = soil_advice(crop, last["N"], last["P"], last["K"], last["ph"])
    if crop in CROP_ECONOMICS:
        y = predict_yield_one(crop, last)
        _, price, cost = CROP_ECONOMICS[crop]
        econ = {"yield": y, "revenue": y * 1000 * price, "cost": cost, "profit": y * 1000 * price - cost}
    return build_report_html(last, items, econ)


def scan_cities(cities, soil: dict, model):
    """Fetches live weather for each city and predicts the best crop with the given soil values."""
    rows, errors = [], []
    for c in cities:
        try:
            w = fetch_weather(c)
        except Exception as e:
            errors.append(f"{c}: {e}")
            continue
        x = pd.DataFrame([{**soil, "temperature": w["temperature"], "humidity": w["humidity"],
                           "rainfall": float(np.clip(w["rainfall"], 20, 300))}])[FEATURES]
        proba = model.predict_proba(x)[0]
        i = int(np.argmax(proba))
        rows.append({
            "City": c.title(), "Temp (°C)": round(w["temperature"], 1), "Humidity (%)": round(w["humidity"], 0),
            "Rainfall 30d (mm)": round(w["rainfall"], 0), "Recommended crop": model.classes_[i],
            "Confidence": float(proba[i]), "lat": w["lat"], "lon": w["lon"],
        })
    return pd.DataFrame(rows), errors


# ----------------------------------------------------------------------------
# 3b. Live weather fetch (Open-Meteo)
# ----------------------------------------------------------------------------
def fetch_weather(city: str):
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}, timeout=10,
    ).json()
    if not geo.get("results"):
        raise ValueError(f"City '{city}' not found.")
    place = geo["results"][0]

    w = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m",
            "daily": "precipitation_sum",
            "past_days": 30,
            "forecast_days": 1,
            "timezone": "auto",
        },
        timeout=10,
    ).json()

    rain = sum(v for v in w["daily"]["precipitation_sum"] if v is not None)
    return {
        "place": f"{place['name']}, {place.get('admin1', '')}, {place.get('country', '')}",
        "lat": float(place["latitude"]),
        "lon": float(place["longitude"]),
        "temperature": float(w["current"]["temperature_2m"]),
        "humidity": float(w["current"]["relative_humidity_2m"]),
        "rainfall": float(rain),
    }


def apply_weather_to_inputs():
    """Button callback: runs BEFORE widgets are drawn, so it can safely set widget values."""
    try:
        info = fetch_weather(st.session_state["city_input"])
        st.session_state["in_temp"] = float(np.clip(info["temperature"], 8.0, 44.0))
        st.session_state["in_hum"] = float(np.clip(info["humidity"], 10.0, 100.0))
        st.session_state["in_rain"] = float(np.clip(info["rainfall"], 20.0, 300.0))
        st.session_state["weather_msg"] = ("ok", f"Live weather loaded for {info['place']}")
        st.session_state["geo"] = (info["lat"], info["lon"])
    except Exception as e:
        st.session_state["weather_msg"] = ("err", f"Could not fetch weather: {e}")


# ----------------------------------------------------------------------------
# 4. SIDEBAR - DATA SOURCE + RETRAIN
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 3b. BIG FEATURES - yield model, profit, soil advisor, forecast, history
# ----------------------------------------------------------------------------
# ILLUSTRATIVE assumptions only (editable in the app): typical yield t/ha, price Rs/kg, cost Rs/ha
CROP_ECONOMICS = {
    "rice": (4.0, 23, 90000), "maize": (3.0, 21, 60000), "chickpea": (1.2, 55, 45000),
    "kidneybeans": (1.0, 90, 50000), "pigeonpeas": (1.0, 70, 45000), "mothbeans": (0.5, 70, 30000),
    "mungbean": (0.8, 75, 35000), "blackgram": (0.8, 70, 35000), "lentil": (1.0, 60, 40000),
    "pomegranate": (12, 70, 250000), "banana": (35, 15, 300000), "mango": (8, 50, 150000),
    "grapes": (20, 60, 400000), "watermelon": (25, 12, 120000), "muskmelon": (18, 20, 120000),
    "apple": (10, 100, 400000), "orange": (15, 40, 200000), "papaya": (40, 15, 250000),
    "coconut": (8, 30, 150000), "cotton": (2.0, 65, 100000), "jute": (2.5, 50, 70000),
    "coffee": (1.5, 250, 200000),
}
ACRE_PER_HA = 2.471
YIELD_PATH = os.path.join(MODEL_DIR, "yield_model_rf.joblib")  # new name so an older saved model is not reused
HISTORY_PATH = os.path.join(MODEL_DIR, "prediction_history.csv")
CLIP_LO = np.array([0, 5, 5, 8, 10, 3.5, 20], dtype=float)
CLIP_HI = np.array([140, 145, 205, 44, 100, 9.9, 300], dtype=float)


def _profile_stds(p):
    n, pp, k, _, _, _, r = p
    return np.array([max(5, 0.12 * n), max(4, 0.12 * pp), max(4, 0.10 * k), 2.0, 4.0, 0.4, max(8, 0.15 * r)])


@st.cache_data
def generate_yield_dataset(samples_per_crop: int = 250) -> pd.DataFrame:
    """Synthetic yield data: yield drops as conditions move away from the crop's ideal."""
    rng = np.random.default_rng(7)
    rows = []
    for crop, p in CROP_PROFILES.items():
        mu, sd = np.array(p, dtype=float), _profile_stds(p)
        draws = np.clip(rng.normal(mu, sd * 1.8, size=(samples_per_crop, 7)), CLIP_LO, CLIP_HI)
        z = np.abs((draws - mu) / sd).mean(axis=1)
        factor = np.clip(1.12 - 0.13 * z, 0.3, 1.12)
        y = CROP_ECONOMICS[crop][0] * factor * rng.normal(1, 0.04, samples_per_crop)
        for d, yy in zip(draws, y):
            rows.append([crop, *d, yy])
    return pd.DataFrame(rows, columns=["crop"] + FEATURES + ["yield_t_ha"]).round(3)


def train_yield_model() -> dict:
    data = generate_yield_dataset()
    X, y = data[["crop"] + FEATURES], data["yield_t_ha"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = Pipeline([
        ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["crop"])],
                                   remainder="passthrough")),
        ("reg", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    out = {
        "model": pipe,
        "r2": r2_score(y_te, pred),
        "mae": mean_absolute_error(y_te, pred),
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "n": len(data),
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(out, YIELD_PATH)
    return out


@st.cache_resource(show_spinner="Training yield-prediction model automatically...")
def get_yield_bundle(retrain_token: int) -> dict:
    if retrain_token == 0 and os.path.exists(YIELD_PATH):
        try:
            return joblib.load(YIELD_PATH)
        except Exception:
            pass
    return train_yield_model()


def soil_advice(crop: str, n: float, p: float, k: float, ph: float):
    """Rule-based soil/fertilizer guidance versus the crop's typical conditions."""
    tn, tp, tk, _, _, tph, _ = CROP_PROFILES[crop]
    items = []
    for label, val, typ, product, frac in [
        ("Nitrogen (N)", n, tn, "Urea", 0.46),
        ("Phosphorus (P)", p, tp, "DAP / SSP", 0.46),
        ("Potassium (K)", k, tk, "Muriate of Potash", 0.60),
    ]:
        diff = val - typ
        if diff < -0.10 * typ:
            need = abs(diff) / frac
            items.append((label, val, typ, "Low", "#e53935",
                          f"Add about {need:.0f} kg/ha of {product} to cover the {abs(diff):.0f} kg/ha gap."))
        elif diff > 0.25 * typ:
            items.append((label, val, typ, "High", "#fb8c00",
                          "Already above typical need - skip this fertilizer and avoid over-application."))
        else:
            items.append((label, val, typ, "Adequate", "#43a047", "Level is fine for this crop."))
    d = ph - tph
    if d < -0.5:
        items.append(("Soil pH", ph, tph, "Acidic", "#e53935",
                      "Soil is acidic for this crop - apply agricultural lime and organic compost."))
    elif d > 0.5:
        items.append(("Soil pH", ph, tph, "Alkaline", "#e53935",
                      "Soil is alkaline for this crop - use gypsum or elemental sulphur and add organic matter."))
    else:
        items.append(("Soil pH", ph, tph, "Adequate", "#43a047", "pH is in a good range for this crop."))
    score = np.mean([
        max(0, 1 - abs(n - tn) / tn), max(0, 1 - abs(p - tp) / tp),
        max(0, 1 - abs(k - tk) / tk), max(0, 1 - abs(ph - tph) / 1.5),
    ]) * 100
    return items, float(score)


def fetch_forecast(city: str):
    geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                       params={"name": city, "count": 1}, timeout=10).json()
    if not geo.get("results"):
        raise ValueError(f"City '{city}' not found.")
    pl = geo["results"][0]
    d = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": pl["latitude"], "longitude": pl["longitude"],
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "forecast_days": 7, "timezone": "auto"},
        timeout=10,
    ).json()["daily"]
    fdf = pd.DataFrame({"Date": d["time"], "Max °C": d["temperature_2m_max"],
                        "Min °C": d["temperature_2m_min"], "Rain mm": d["precipitation_sum"]}).fillna(0)
    name = f"{pl['name']}, {pl.get('admin1', '')}, {pl.get('country', '')}"
    return name, pl["latitude"], pl["longitude"], fdf


def log_prediction(row: dict):
    os.makedirs(MODEL_DIR, exist_ok=True)
    pd.DataFrame([row]).to_csv(HISTORY_PATH, mode="a", header=not os.path.exists(HISTORY_PATH), index=False)


def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_PATH):
        try:
            return pd.read_csv(HISTORY_PATH)
        except Exception:
            pass
    return pd.DataFrame()


if "retrain_token" not in st.session_state:
    st.session_state["retrain_token"] = 0

st.sidebar.markdown(
    '<div class="brand"><div class="logo">🌾</div><div><div class="t">AgriYield AI</div>'
    '<div class="s">Smart crop recommendation</div></div></div>',
    unsafe_allow_html=True,
)
st.sidebar.subheader("⚙️ Dataset & Training")
source = st.sidebar.radio("Training data", ["Built-in synthetic dataset", "Upload my own CSV"])

df = generate_builtin_dataset()
if source == "Upload my own CSV":
    up = st.sidebar.file_uploader("CSV with N, P, K, temperature, humidity, ph, rainfall, label", type="csv")
    if up is not None:
        try:
            df = normalise_uploaded_csv(pd.read_csv(up))
            st.sidebar.success(f"Loaded {len(df)} rows")
        except Exception as e:
            st.sidebar.error(str(e))
    else:
        st.sidebar.info("No file uploaded - using the built-in dataset for now.")

if st.sidebar.button("Force re-train now"):
    st.session_state["retrain_token"] += 1

bundle = get_bundle(dataset_signature(df), df, st.session_state["retrain_token"])
models, metrics = bundle["models"], bundle["metrics"]
yield_bundle = get_yield_bundle(st.session_state["retrain_token"])
st.sidebar.success(f"Model ready: {bundle['best_name']}")
st.sidebar.caption(f"Train rows: {bundle['n_train']} | Test rows: {bundle['n_test']}")


# ----------------------------------------------------------------------------
# 5. MAIN UI
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="navbar">
        <div class="brand-n">🌾 AgriYield AI</div>
        <div class="right">
            <span class="pill ghost">Weather + Soil</span>
            <span class="pill ghost">Auto-trained ML</span>
            <span class="pill">● Model ready: {bundle['best_name']}</span>
        </div>
    </div>
    <div class="hero2">
        <div>
            <h1>Grow the <span>right crop</span>,<br>every season.</h1>
            <p>Enter your soil nutrients and local weather - our machine learning model instantly
            recommends the most suitable crop for your field. No setup, it trains itself.</p>
            <span class="cta">👇 Try the predictor below</span>
        </div>
        <div class="float-wrap">
            <span class="f" style="left:8%; top:10%; animation-delay:0s;">🌾</span>
            <span class="f" style="left:55%; top:2%; animation-delay:.8s;">🌽</span>
            <span class="f" style="left:30%; top:42%; animation-delay:1.6s;">🍇</span>
            <span class="f" style="left:70%; top:48%; animation-delay:2.4s;">🥭</span>
            <span class="f" style="left:6%; top:72%; animation-delay:3.2s;">🍌</span>
            <span class="f" style="left:52%; top:76%; animation-delay:4s;">🍊</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

best_row = metrics.iloc[0]
k1, k2, k3, k4 = st.columns(4)
kpi(k1, bundle["best_name"], "Best model")
kpi(k2, f"{best_row['Accuracy']:.1%}", "Test accuracy")
kpi(k3, str(len(bundle["classes"])), "Crops supported")
kpi(k4, f"{len(df):,}", "Training samples")

st.markdown(
    """
    <div class="title-block"><h2>Why AgriYield AI?</h2><p>Smart farming decisions powered by data</p><div class="bar"></div></div>
    <div class="feat-grid">
        <div class="feat"><div class="ic">🌦️</div><h4>Live Weather</h4><p>Fetch real temperature, humidity and rainfall for your city in one click.</p></div>
        <div class="feat"><div class="ic">🧪</div><h4>Soil Analysis</h4><p>Uses Nitrogen, Phosphorus, Potassium and pH to understand your soil.</p></div>
        <div class="feat"><div class="ic">🤖</div><h4>Auto-Trained AI</h4><p>The model trains itself on first run and re-trains when data changes.</p></div>
        <div class="feat"><div class="ic">📊</div><h4>5-Model Comparison</h4><p>Accuracy, Precision, Recall and F1 compared to pick the best algorithm.</p></div>
        <div class="feat"><div class="ic">💰</div><h4>Yield &amp; Profit</h4><p>A second ML model estimates yield and profit per hectare for every crop.</p></div>
        <div class="feat"><div class="ic">🔬</div><h4>What-If Analysis</h4><p>See how changing one input flips the recommended crop.</p></div>
        <div class="feat"><div class="ic">🛰️</div><h4>Regional Scan</h4><p>Compare best crops across many cities on a live-weather map.</p></div>
        <div class="feat"><div class="ic">🤖</div><h4>Farming Assistant</h4><p>Ask why a crop was chosen, which fertilizer to use, and more.</p></div>
        <div class="feat"><div class="ic">💰</div><h4>Yield &amp; Profit</h4><p>A second ML model predicts yield and estimates profit per hectare and acre.</p></div>
        <div class="feat"><div class="ic">🧪</div><h4>Fertilizer Advisor</h4><p>Soil health score with simple fertilizer and pH correction advice.</p></div>
        <div class="feat"><div class="ic">🗺️</div><h4>Forecast &amp; Map</h4><p>7-day forecast, location map and farm alerts for rain, heat and cold.</p></div>
        <div class="feat"><div class="ic">🕘</div><h4>Prediction History</h4><p>Every prediction is saved, charted and downloadable as CSV.</p></div>
    </div>
    <div class="title-block"><h2>How it works</h2><p>Three simple steps</p><div class="bar"></div></div>
    <div class="steps">
        <div class="step"><div class="n">1</div><div><h4>Enter weather</h4><p>Type values or fetch live weather by city name.</p></div></div>
        <div class="step"><div class="n">2</div><div><h4>Enter soil data</h4><p>Add N, P, K and pH from your soil test report.</p></div></div>
        <div class="step"><div class="n">3</div><div><h4>Get recommendation</h4><p>See the best crop, top 3 options and how your field compares.</p></div></div>
    </div>
    <div class="title-block"><h2>🚀 Try the Crop Predictor</h2><p>Fill in the details and let the model decide</p><div class="bar"></div></div>
    """,
    unsafe_allow_html=True,
)

(tab_predict, tab_profit, tab_soil, tab_map, tab_whatif, tab_region, tab_chat,
 tab_hist, tab_compare, tab_data) = st.tabs([
    "🌱 Crop Predictor", "💰 Yield & Profit", "🧪 Soil Advisor", "🗺️ Weather & Map",
    "🔬 What-If", "🛰️ Regional Scan", "🤖 Assistant",
    "🕘 History", "📊 Model Comparison", "🗂️ Data Explorer",
])

# ---------------- TAB 1: PREDICTOR ----------------
with tab_predict:
    section("1", "🌦️ Weather data")
    with st.container(border=True):
        wc1, wc2 = st.columns([3, 1])
        wc1.text_input("📍 City (to fetch live weather)", value="Delhi", key="city_input")
        wc2.write("")
        wc2.write("")
        wc2.button("📡 Fetch live weather", on_click=apply_weather_to_inputs, use_container_width=True)
        if "weather_msg" in st.session_state:
            kind, msg = st.session_state["weather_msg"]
            (st.success if kind == "ok" else st.error)(msg)

        w1, w2, w3 = st.columns(3)
        temperature = w1.number_input("🌡️ Temperature (°C)", 8.0, 44.0, 25.0, 0.1, key="in_temp")
        humidity = w2.number_input("💧 Humidity (%)", 10.0, 100.0, 70.0, 0.5, key="in_hum")
        rainfall = w3.number_input("🌧️ Rainfall (mm, last 30 days / season)", 20.0, 300.0, 100.0, 1.0, key="in_rain")

    section("2", "🧪 Soil data")
    with st.container(border=True):
        s1, s2, s3, s4 = st.columns(4)
        N = s1.number_input("🟦 Nitrogen N (kg/ha)", 0, 140, 60)
        P = s2.number_input("🟧 Phosphorus P (kg/ha)", 5, 145, 50)
        K = s3.number_input("🟪 Potassium K (kg/ha)", 5, 205, 45)
        ph = s4.number_input("⚗️ Soil pH", 3.5, 9.9, 6.5, 0.1)

    section("3", "🤖 Algorithm")
    algo_choice = st.selectbox(
        "Algorithm",
        ["Best model (auto-selected)"] + list(models.keys()),
        label_visibility="collapsed",
    )

    if st.button("🌾 Recommend Crop", type="primary", use_container_width=True):
        chosen = bundle["best_name"] if algo_choice.startswith("Best") else algo_choice
        model = models[chosen]
        x = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURES)

        proba = model.predict_proba(x)[0]
        order = np.argsort(proba)[::-1][:3]
        classes = model.classes_

        top_crop = classes[order[0]]
        log_prediction({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "N": N, "P": P, "K": K,
            "Temperature": temperature, "Humidity": humidity, "pH": ph, "Rainfall": rainfall,
            "Model": chosen, "Recommended crop": top_crop, "Confidence": round(float(proba[order[0]]), 4),
        })
        st.session_state["last"] = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "crop": top_crop,
            "confidence": float(proba[order[0]]), "model": chosen,
            "N": N, "P": P, "K": K, "temperature": temperature, "humidity": humidity, "ph": ph,
            "rainfall": rainfall, "top3_list": [(classes[i], float(proba[i])) for i in order],
        }
        st.markdown(
            f"""
            <div class="result">
                <div class="emoji">{CROP_EMOJI.get(top_crop, "🌱")}</div>
                <div class="sub">Recommended crop for your field</div>
                <div class="name">{top_crop.title()}</div>
                <div class="sub">Confidence {proba[order[0]]:.1%} &nbsp;|&nbsp; Model: {chosen}</div>
                <div class="meter"><div style="width:{proba[order[0]] * 100:.0f}%"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        section("4", "🏆 Top 3 suggestions")
        top = pd.DataFrame({
            "Crop": [f"{CROP_EMOJI.get(classes[i], '🌱')} {classes[i].title()}" for i in order],
            "Confidence": [proba[i] for i in order],
        })
        c1, c2 = st.columns([1, 2])
        c1.dataframe(top.style.format({"Confidence": "{:.1%}"}), use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(6, 2.5))
        plain = top["Crop"].str.split(" ", n=1).str[1]
        sns.barplot(x=top["Confidence"], y=plain, hue=plain, legend=False,
                    palette=["#1b5e20", "#43a047", "#a5d6a7"], ax=ax)
        ax.set_ylabel("Crop")
        ax.set_xlim(0, 1)
        ax.set_title("Top 3 crop suggestions")
        c2.pyplot(fig)

        # your field vs typical conditions for the recommended crop
        if top_crop in CROP_PROFILES:
            section("5", f"📋 Your field vs typical conditions for {top_crop.title()}")
            labels = ["Nitrogen N", "Phosphorus P", "Potassium K", "Temperature (°C)",
                      "Humidity (%)", "Soil pH", "Rainfall (mm)"]
            yours = [N, P, K, temperature, humidity, ph, rainfall]
            ideal = CROP_PROFILES[top_crop]
            compare_df = pd.DataFrame({
                "Parameter": labels,
                "Your field": yours,
                "Typical for crop": ideal,
            })
            compare_df["Difference"] = compare_df["Your field"] - compare_df["Typical for crop"]
            st.dataframe(
                compare_df.style.format({"Your field": "{:.1f}", "Typical for crop": "{:.1f}", "Difference": "{:+.1f}"})
                .background_gradient(subset=["Difference"], cmap="RdYlGn_r", vmin=-60, vmax=60),
                use_container_width=True, hide_index=True,
            )
            st.caption("Typical values are the average growing conditions used to build the built-in dataset.")

        # warn if inputs are outside the training data
        out = [f for f, v in zip(FEATURES, x.iloc[0]) if not (bundle["feature_ranges"][f][0] <= v <= bundle["feature_ranges"][f][1])]
        if out:
            st.warning(f"These inputs are outside the training data range, so the result may be less reliable: {', '.join(out)}")

# Values from the Crop Predictor tab (current inputs) used by the tabs below
x_now = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURES)
best_model = models[bundle["best_name"]]
suit_now = dict(zip(best_model.classes_, best_model.predict_proba(x_now)[0]))
current_top = max(suit_now, key=suit_now.get)

# ---------------- TAB: YIELD & PROFIT ----------------
with tab_profit:
    section("💰", "Expected yield and profit for your field")
    st.caption("Uses the inputs from the Crop Predictor tab. A second ML model (regression) trained "
               "automatically predicts yield. Prices and costs are illustrative assumptions - edit them below.")
    m1, m2, m3 = st.columns(3)
    kpi(m1, f"{yield_bundle['r2']:.1%}", "Yield model R² score")
    kpi(m2, f"{yield_bundle['mae']:.2f} t/ha", "Mean absolute error")
    kpi(m3, f"{yield_bundle['n']:,}", "Yield training samples")
    st.write("")

    econ_df = pd.DataFrame([{"Crop": c.title(), "Price (₹/kg)": float(v[1]), "Cost (₹/ha)": float(v[2])}
                            for c, v in CROP_ECONOMICS.items()])
    with st.expander("⚙️ Edit price and cost assumptions"):
        econ_edit = st.data_editor(econ_df, hide_index=True, use_container_width=True,
                                   disabled=["Crop"], key="econ_editor")
    econ_map = {r["Crop"].lower(): (r["Price (₹/kg)"], r["Cost (₹/ha)"]) for _, r in econ_edit.iterrows()}

    crops_list = list(CROP_PROFILES)
    x_all = pd.DataFrame([[c, N, P, K, temperature, humidity, ph, rainfall] for c in crops_list],
                         columns=["crop"] + FEATURES)
    y_pred = yield_bundle["model"].predict(x_all)

    rows = []
    for c, yv in zip(crops_list, y_pred):
        price, cost = econ_map[c]
        revenue = yv * 1000 * price
        profit = revenue - cost
        rows.append({
            "Crop": f"{CROP_EMOJI.get(c, '🌱')} {c.title()}",
            "Suitability": suit_now.get(c, 0.0),
            "Yield (t/ha)": yv,
            "Revenue (₹/ha)": revenue,
            "Cost (₹/ha)": cost,
            "Profit (₹/ha)": profit,
            "Profit (₹/acre)": profit / ACRE_PER_HA,
            "Score": suit_now.get(c, 0.0) * profit,
        })
    prof_df = pd.DataFrame(rows).sort_values("Profit (₹/ha)", ascending=False).reset_index(drop=True)
    pick = prof_df.sort_values("Score", ascending=False).iloc[0]
    top_profit = prof_df.iloc[0]

    c_a, c_b = st.columns(2)
    c_a.success(f"💎 Highest profit: **{top_profit['Crop']}** - about ₹{top_profit['Profit (₹/ha)']:,.0f} per hectare")
    c_b.info(f"🏅 Best balanced pick (suitability × profit): **{pick['Crop']}** - "
             f"yield {pick['Yield (t/ha)']:.1f} t/ha, profit ₹{pick['Profit (₹/ha)']:,.0f}/ha")

    show = prof_df.drop(columns=["Score"])
    st.dataframe(
        show.style.format({"Suitability": "{:.1%}", "Yield (t/ha)": "{:.2f}", "Revenue (₹/ha)": "₹{:,.0f}",
                           "Cost (₹/ha)": "₹{:,.0f}", "Profit (₹/ha)": "₹{:,.0f}", "Profit (₹/acre)": "₹{:,.0f}"})
        .background_gradient(subset=["Suitability"], cmap="Greens"),
        use_container_width=True, hide_index=True,
    )

    top8 = prof_df.head(8).iloc[::-1]
    fig_p, ax_p = plt.subplots(figsize=(8, 4))
    names = top8["Crop"].str.split(" ", n=1).str[1]
    ax_p.barh(names, top8["Profit (₹/ha)"], color=["#2e7d32" if v >= 0 else "#e53935" for v in top8["Profit (₹/ha)"]])
    ax_p.set_xlabel("Estimated profit (Rs per hectare)")
    ax_p.set_title("Top 8 crops by estimated profit")
    st.pyplot(fig_p)
    st.caption("⚠️ Estimates only. Real profit depends on market price, input costs, irrigation and weather risk.")

# ---------------- TAB: SOIL ADVISOR ----------------
with tab_soil:
    section("🧪", "Soil health and fertilizer advisor")
    st.caption("Compares your soil values (from the Crop Predictor tab) with typical needs of the chosen crop.")
    default_idx = list(CROP_PROFILES).index(current_top) if current_top in CROP_PROFILES else 0
    crop_sel = st.selectbox("Crop you want to grow", list(CROP_PROFILES), index=default_idx,
                            format_func=lambda c: f"{CROP_EMOJI.get(c, '🌱')} {c.title()}")
    items, score = soil_advice(crop_sel, N, P, K, ph)

    st.markdown(f"**Soil health score for {crop_sel.title()}: {score:.0f} / 100**")
    st.progress(int(min(100, max(0, score))))

    cols = st.columns(4)
    for col, (label, val, typ, status, color, tip) in zip(cols, items):
        col.markdown(
            f"""<div style="border-radius:16px; padding:1rem; border:2px solid {color};
                background:{color}14; min-height:190px;">
                <div style="font-weight:700;">{label}</div>
                <div style="font-size:1.6rem; font-weight:800; color:{color};">{val:g}</div>
                <div style="font-size:.8rem; opacity:.75;">Typical: {typ:g}</div>
                <div style="margin:.4rem 0; font-weight:700; color:{color};">● {status}</div>
                <div style="font-size:.85rem;">{tip}</div></div>""",
            unsafe_allow_html=True,
        )
    st.caption("General guidance based on typical crop needs. For exact doses, get a soil test and "
               "consult your local agriculture office (KVK).")

# ---------------- TAB: WEATHER & MAP ----------------
with tab_map:
    section("🗺️", "7-day forecast and location map")
    fc_city = st.session_state.get("city_input", "Delhi")
    st.write(f"City: **{fc_city}** (change it in the Crop Predictor tab)")
    if st.button("📡 Load 7-day forecast"):
        try:
            st.session_state["forecast"] = fetch_forecast(fc_city)
        except Exception as e:
            st.session_state.pop("forecast", None)
            st.error(f"Could not load forecast: {e}")

    if "forecast" in st.session_state:
        f_name, f_lat, f_lon, fdf = st.session_state["forecast"]
        st.success(f"Forecast for {f_name}")
        st.map(pd.DataFrame({"lat": [f_lat], "lon": [f_lon]}))

        fig_f, ax_f = plt.subplots(figsize=(9, 3.8))
        ax_f.bar(fdf["Date"], fdf["Rain mm"], color="#64b5f6", label="Rain (mm)")
        ax_f.set_ylabel("Rain (mm)")
        ax_t = ax_f.twinx()
        ax_t.plot(fdf["Date"], fdf["Max °C"], color="#e53935", marker="o", label="Max °C")
        ax_t.plot(fdf["Date"], fdf["Min °C"], color="#1e88e5", marker="o", label="Min °C")
        ax_t.set_ylabel("Temperature (°C)")
        ax_t.legend(loc="upper right")
        ax_f.set_title("Next 7 days")
        plt.setp(ax_f.get_xticklabels(), rotation=30)
        st.pyplot(fig_f)

        section("⚠️", "Farm alerts for the coming week")
        total_rain, max_t, min_t = fdf["Rain mm"].sum(), fdf["Max °C"].max(), fdf["Min °C"].min()
        alerts = []
        if total_rain > 80:
            alerts.append(("🌧️ Heavy rain expected", f"About {total_rain:.0f} mm in 7 days. Ensure drainage and delay fertilizer application."))
        elif total_rain < 5:
            alerts.append(("☀️ Dry week", f"Only {total_rain:.0f} mm rain expected. Plan irrigation."))
        if max_t >= 38:
            alerts.append(("🔥 Heat stress risk", f"Max temperature up to {max_t:.0f}°C. Irrigate in early morning or evening."))
        if min_t <= 8:
            alerts.append(("❄️ Cold / frost risk", f"Min temperature down to {min_t:.0f}°C. Protect young plants."))
        if not alerts:
            alerts.append(("✅ Conditions look normal", "No extreme weather in the 7-day forecast."))
        for title, text in alerts:
            st.warning(f"**{title}** - {text}")
        st.dataframe(fdf, use_container_width=True, hide_index=True)
    else:
        st.info("Click the button to load the forecast and see your location on the map.")

# ---------------- TAB: HISTORY ----------------
with tab_hist:
    section("🕘", "Prediction history")
    hist = load_history()
    if hist.empty:
        st.info("No predictions yet. Go to the Crop Predictor tab and click 'Recommend Crop'.")
    else:
        h1, h2, h3 = st.columns(3)
        kpi(h1, str(len(hist)), "Total predictions")
        kpi(h2, str(hist["Recommended crop"].mode().iloc[0]).title(), "Most recommended crop")
        kpi(h3, f"{hist['Confidence'].mean():.1%}", "Average confidence")
        st.write("")
        st.dataframe(hist.iloc[::-1], use_container_width=True, hide_index=True)

        counts = hist["Recommended crop"].value_counts().head(10).iloc[::-1]
        fig_h, ax_h = plt.subplots(figsize=(7, 3.5))
        ax_h.barh(counts.index.str.title(), counts.values, color="#43a047")
        ax_h.set_xlabel("Times recommended")
        ax_h.set_title("Most recommended crops")
        st.pyplot(fig_h)

        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ Download history (CSV)", hist.to_csv(index=False).encode("utf-8"),
                           "prediction_history.csv", "text/csv", use_container_width=True)
        last_pred = st.session_state.get("last")
        if last_pred:
            d2.download_button("📄 Latest report (HTML)", report_for_last(last_pred).encode("utf-8"),
                               "crop_report.html", "text/html", use_container_width=True,
                               help="Open in a browser, then Print -> Save as PDF")
        else:
            d2.button("📄 Latest report (HTML)", disabled=True, use_container_width=True,
                      help="Run a prediction first")
        if d3.button("🗑️ Clear history", use_container_width=True):
            os.remove(HISTORY_PATH)
            st.rerun()

# ---------------- TAB: WHAT-IF ANALYSIS ----------------
with tab_whatif:
    section("🔬", "What-if analysis")
    st.caption("See how changing ONE input changes the recommendation. All other values come from the Crop Predictor tab.")
    feat = st.selectbox("Parameter to vary", FEATURES, format_func=lambda f: FEATURE_LABELS[f])
    f_lo, f_hi = bundle["feature_ranges"][feat]
    base_vals = {"N": N, "P": P, "K": K, "temperature": temperature, "humidity": humidity,
                 "ph": ph, "rainfall": rainfall}
    grid = np.linspace(f_lo, f_hi, 60)
    sweep = pd.DataFrame([{**base_vals, feat: v} for v in grid])[FEATURES]
    sweep_proba = best_model.predict_proba(sweep)
    cls = best_model.classes_
    base_proba = best_model.predict_proba(pd.DataFrame([base_vals])[FEATURES])[0]
    top_idx = np.argsort(base_proba)[::-1][:3]

    fig_w, ax_w = plt.subplots(figsize=(9, 4))
    for i in top_idx:
        ax_w.plot(grid, sweep_proba[:, i], linewidth=2.2, label=cls[i].title())
    ax_w.axvline(base_vals[feat], color="gray", linestyle="--", label="Your value")
    ax_w.set_xlabel(FEATURE_LABELS[feat])
    ax_w.set_ylabel("Model confidence")
    ax_w.set_ylim(0, 1.02)
    ax_w.set_title(f"How {FEATURE_LABELS[feat]} changes the top suggestions")
    ax_w.legend()
    st.pyplot(fig_w)

    best_idx = sweep_proba.argmax(axis=1)
    st.markdown("**Recommended crop as the value changes:**")
    seg_start = 0
    for j in range(1, len(grid) + 1):
        if j == len(grid) or best_idx[j] != best_idx[seg_start]:
            c_name = cls[best_idx[seg_start]]
            st.write(f"- {CROP_EMOJI.get(c_name, '🌱')} **{c_name.title()}**: "
                     f"{grid[seg_start]:.1f} to {grid[j - 1]:.1f}")
            seg_start = j

# ---------------- TAB: REGIONAL SCAN ----------------
with tab_region:
    section("🛰️", "Regional crop scan")
    st.caption("Enter several cities. The app fetches live weather for each one and recommends the best crop "
               "using the soil values from the Crop Predictor tab. Needs an internet connection.")
    cities_text = st.text_area("Cities (comma-separated, max 10)", "Delhi, Mumbai, Chennai, Kolkata, Jaipur")
    if st.button("🛰️ Scan cities"):
        city_list = [c.strip() for c in cities_text.split(",") if c.strip()][:10]
        with st.spinner("Fetching live weather for each city..."):
            st.session_state["scan"] = scan_cities(city_list, {"N": N, "P": P, "K": K, "ph": ph}, best_model)
    if "scan" in st.session_state:
        scan_df, scan_errs = st.session_state["scan"]
        for err in scan_errs:
            st.warning(err)
        if not scan_df.empty:
            show_scan = scan_df.drop(columns=["lat", "lon"]).copy()
            show_scan["Recommended crop"] = show_scan["Recommended crop"].map(
                lambda c: f"{CROP_EMOJI.get(c, '🌱')} {c.title()}")
            st.dataframe(show_scan.style.format({"Confidence": "{:.1%}"}), use_container_width=True, hide_index=True)
            st.map(scan_df.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude"]])
            st.markdown("**Crops recommended across these cities**")
            st.bar_chart(scan_df["Recommended crop"].value_counts())

# ---------------- TAB: FARMING ASSISTANT ----------------
with tab_chat:
    section("🤖", "Farming assistant")
    st.caption("Offline rule-based assistant. Ask about fertilizer, water, season, soil pH, yield, profit or pests. "
               "Run a prediction first for personalised answers.")
    if "chat" not in st.session_state:
        st.session_state["chat"] = [{"role": "assistant",
                                     "content": "Namaste! 🌾 Ask me things like *why this crop?* or *which fertilizer?*"}]
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    question = st.chat_input("Ask a farming question...")
    if question:
        answer = assistant_reply(question, st.session_state.get("last"))
        st.session_state["chat"] += [{"role": "user", "content": question},
                                     {"role": "assistant", "content": answer}]
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)

# ---------------- TAB 2: MODEL COMPARISON ----------------
with tab_compare:
    section("📊", "Algorithm performance (held-out 20% test data)")
    st.dataframe(
        metrics.style.format({c: "{:.2%}" for c in ["Accuracy", "Precision", "Recall", "F1 Score"]})
        .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1 Score"], color="#c8e6c9"),
        use_container_width=True, hide_index=True,
    )
    st.info(f"Best model (highest F1): **{bundle['best_name']}**")

    long = metrics.melt(id_vars="Algorithm", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=long, x="Algorithm", y="Score", hue="Metric", ax=ax)
    ax.set_ylim(0.8, 1.01)
    ax.set_title("Accuracy / Precision / Recall / F1 comparison")
    plt.xticks(rotation=15)
    st.pyplot(fig)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Confusion matrix (best model)**")
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(bundle["confusion_matrix"], cmap="Greens", ax=ax_cm,
                    xticklabels=bundle["classes"], yticklabels=bundle["classes"], cbar=False)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)
    with col_b:
        st.markdown("**Feature importance (Random Forest)**")
        imp = pd.Series(models["Random Forest"].feature_importances_, index=FEATURES).sort_values()
        fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
        imp.plot(kind="barh", color="darkgreen", ax=ax_imp)
        ax_imp.set_xlabel("Importance")
        st.pyplot(fig_imp)

# ---------------- TAB 3: DATA EXPLORER ----------------
with tab_data:
    section("🗂️", "Dataset preview")
    st.write(f"{len(df)} rows | {df[TARGET].nunique()} crops")
    st.dataframe(df.head(20), use_container_width=True)

    d1, d2 = st.columns(2)
    with d1:
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="YlGn", ax=ax1)
        ax1.set_title("Feature correlation")
        st.pyplot(fig1)
    with d2:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=df, x="temperature", y="rainfall", hue=TARGET, ax=ax2, legend=False, s=15)
        ax2.set_title("Temperature vs Rainfall (coloured by crop)")
        st.pyplot(fig2)

st.markdown(
    """
    <div class="footer2">
        <div><h4>🌾 AgriYield AI</h4><p>Smart crop recommendation using weather and soil data. A machine learning mini project that trains itself automatically.</p></div>
        <div><h4>Tech stack</h4><ul><li>🐍 Python</li><li>🎈 Streamlit</li><li>🧠 scikit-learn</li><li>🌐 Open-Meteo weather API</li></ul></div>
        <div><h4>Features</h4><ul><li>✔ Crop recommendation</li><li>✔ Yield &amp; profit estimate</li><li>✔ What-if &amp; regional scan</li><li>✔ Report download</li></ul></div>
    </div>
    <div class="copy">Built with 💚 for farmers | Mini Project</div>
    """,
    unsafe_allow_html=True,
)
