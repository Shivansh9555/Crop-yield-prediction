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

st.set_page_config(page_title="AgriYield AI - Crop Recommendation", page_icon="🌾", layout="wide")

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
# 3. LIVE WEATHER (Open-Meteo, free, no API key)
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
    except Exception as e:
        st.session_state["weather_msg"] = ("err", f"Could not fetch weather: {e}")


# ----------------------------------------------------------------------------
# 4. SIDEBAR - DATA SOURCE + RETRAIN
# ----------------------------------------------------------------------------
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
st.sidebar.success(f"Model ready: {bundle['best_name']}")
st.sidebar.caption(f"Train rows: {bundle['n_train']} | Test rows: {bundle['n_test']}")


# ----------------------------------------------------------------------------
# 5. MAIN UI
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🌾 AgriYield AI</h1>
        <p>Smart crop recommendation from soil and weather data - the model trains itself automatically.</p>
        <span class="badge">🌡️ Live weather</span>
        <span class="badge">🧪 Soil N-P-K &amp; pH</span>
        <span class="badge">🤖 Auto-training</span>
        <span class="badge">📊 5-model comparison</span>
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
st.write("")

tab_predict, tab_compare, tab_data = st.tabs(["🌱 Crop Predictor", "📊 Model Comparison", "🗂️ Data Explorer"])

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
    '<div class="footer">🌱 <b>AgriYield AI</b> - Crop recommendation using weather &amp; soil data '
    '| Built with Streamlit &amp; scikit-learn | Mini Project</div>',
    unsafe_allow_html=True,
)
