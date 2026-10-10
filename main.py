import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="AgriYield AI - Crop Prediction Platform",
    page_icon="🌾",
    layout="wide"
)

# 2. SYNTHETIC DUMMY DATA GENERATOR
@st.cache_data
def generate_agricultural_data():
    np.random.seed(42)
    n_samples = 1200
    
    crops = ['Wheat', 'Rice', 'Maize', 'Cotton', 'Sugarcane']
    crop_data = np.random.choice(crops, n_samples)
    
    rainfall = np.random.uniform(400, 1500, n_samples)
    temperature = np.random.uniform(18, 38, n_samples)
    soil_ph = np.random.uniform(5.5, 8.0, n_samples)
    nitrogen = np.random.uniform(20, 140, n_samples)
    phosphorus = np.random.uniform(15, 90, n_samples)
    potassium = np.random.uniform(15, 85, n_samples)
    
    base_yield = 1.5
    crop_multipliers = {'Wheat': 1.2, 'Rice': 1.5, 'Maize': 1.4, 'Cotton': 0.8, 'Sugarcane': 4.5}
    
    yields = []
    for i in range(n_samples):
        c = crop_data[i]
        y = base_yield * crop_multipliers[c]
        y += (rainfall[i] * 0.0015) + (nitrogen[i] * 0.012) + (phosphorus[i] * 0.008)
        y -= abs(temperature[i] - 27) * 0.05
        y -= abs(soil_ph[i] - 6.5) * 0.3
        y += np.random.normal(0, 0.2)
        yields.append(max(0.5, round(y, 2)))

    df = pd.DataFrame({
        'Crop_Type': crop_data,
        'Rainfall_mm': rainfall,
        'Temperature_C': temperature,
        'Soil_pH': soil_ph,
        'Nitrogen_N': nitrogen,
        'Phosphorus_P': phosphorus,
        'Potassium_K': potassium,
        'Yield_Tons_Per_Hectare': yields
    })
    return df

df = generate_agricultural_data()

# 3. TRAINING FUNCTION FOR MULTIPLE MODELS
@st.cache_resource
def train_models(data):
    X = data.drop(columns=['Yield_Tons_Per_Hectare'])
    y = data['Yield_Tons_Per_Hectare']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['Crop_Type'])],
        remainder='passthrough'
    )
    
    # Random Forest
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mse = mean_squared_error(y_test, rf_pred)
    
    # Gradient Boosting
    gb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42))
    ])
    gb_pipeline.fit(X_train, y_train)
    gb_pred = gb_pipeline.predict(X_test)
    gb_r2 = r2_score(y_test, gb_pred)
    gb_mse = mean_squared_error(y_test, gb_pred)
    
    return {
        "Random Forest": {"model": rf_pipeline, "r2": rf_r2, "mse": rf_mse},
        "Gradient Boosting": {"model": gb_pipeline, "r2": gb_r2, "mse": gb_mse}
    }

models = train_models(df)

# 4. FRONTEND STREAMLIT DESIGN
st.title("🌾 Agriculture Crop Yield Assessment Platform")
st.markdown("An end-to-end predictive analytics framework for agricultural optimization.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploratory Data Analysis",
    "⚙️ Model Intelligence Metrics",
    "🔮 Predictive Simulator Engine",
    "📈 Algorithm Comparison"
])

with tab1:
    st.subheader("Dataset Exploratory Insights")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['Yield_Tons_Per_Hectare'], kde=True, color='green', ax=ax)
        ax.set_title("Global Distribution of Crop Yields")
        st.pyplot(fig)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x='Nitrogen_N', y='Yield_Tons_Per_Hectare', hue='Crop_Type', ax=ax2)
        ax2.set_title("Nitrogen Utilization Efficiency Vs Yield Output")
        st.pyplot(fig2)

with tab2:
    st.subheader("Model Validation Metrics")
    for algo, details in models.items():
        col1, col2 = st.columns(2)
        col1.metric(label=f"{algo} R² Score", value=f"{details['r2']:.2%}")
        col2.metric(label=f"{algo} MSE", value=f"{details['mse']:.4f}")
with tab3:
    st.subheader("Instant Simulation Form Tool")
    
    # Algorithm selector
    algo_choice = st.selectbox("Select Algorithm", options=list(models.keys()))
    
    c_col1, c_col2, c_col3 = st.columns(3)
    with c_col1:
        input_crop = st.selectbox("Target Production Crop", options=df['Crop_Type'].unique())
        input_rain = st.slider("Expected Rainfall Context (mm)", 400, 1500, 850)
    with c_col2:
        input_temp = st.slider("Mean Temperature Ambient (°C)", 18, 38, 28)
        input_ph = st.slider("Measured Soil pH Value", 5.5, 8.0, 6.5, step=0.1)
    with c_col3:
        input_n = st.number_input("Nitrogen Fertilizer Mix (kg/ha)", 20, 140, 75)
        input_p = st.number_input("Phosphorus Fertilizer Mix (kg/ha)", 15, 90, 50)
        input_k = st.number_input("Potassium Fertilizer Mix (kg/ha)", 15, 85, 45)

    if st.button("Calculate Yield Prediction", type="primary"):
        input_payload = pd.DataFrame([{
            'Crop_Type': input_crop,
            'Rainfall_mm': input_rain,
            'Temperature_C': input_temp,
            'Soil_pH': input_ph,
            'Nitrogen_N': input_n,
            'Phosphorus_P': input_p,
            'Potassium_K': input_k
        }])
        
        # Use selected algorithm
        selected_model = models[algo_choice]["model"]
        pred = selected_model.predict(input_payload)[0]
        
        st.success(f"### {algo_choice} Prediction → **{pred:.2f} Tons / Hectare**")


with tab4:
    st.subheader("Algorithm Performance Comparison")
    comparison_df = pd.DataFrame({
        "Algorithm": list(models.keys()),
        "R² Score": [details['r2'] for details in models.values()],
        "MSE": [details['mse'] for details in models.values()]
    })
    st.dataframe(comparison_df, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=comparison_df.melt(id_vars="Algorithm", var_name="Metric", value_name="Score"),
                x="Algorithm", y="Score", hue="Metric", ax=ax)
    ax.set_title("Model Performance Comparison")
    st.pyplot(fig)
