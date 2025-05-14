import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------
# Page Setup
# ----------------------
st.set_page_config(layout='wide')
st.title("✈️ Flight Price Prediction App")

# ----------------------
# Data Loading
# ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("all_flights.csv")

    # Basic cleaning
    df['departureTime'] = pd.to_datetime(df['departureTime'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['durationMinutes'] = pd.to_numeric(df['durationTime'], errors='coerce')
    df['carbonEmissionsThisFlight'] = pd.to_numeric(df.get('carbonEmissionsThisFlight'), errors='coerce')
    df['legroom'] = pd.to_numeric(df['legroom'].str.extract(r'(\d+)'), errors='coerce')

    df['weekday'] = df['departureTime'].dt.day_name()
    df['hour'] = df['departureTime'].dt.hour
    df = pd.get_dummies(df, columns=['travelClass', 'weekday'], drop_first=True)

    return df.dropna(subset=['price', 'durationMinutes', 'carbonEmissionsThisFlight', 'legroom'])

df = load_data()

# ----------------------
# Sidebar
# ----------------------
st.sidebar.header("Model Settings")

all_features = [col for col in df.columns if col not in ['price', 'departureTime', 'arrivalAirportTime']]
selected_features = st.sidebar.multiselect("Select features for prediction", all_features, default=[
    'durationMinutes', 'carbonEmissionsThisFlight', 'legroom'
])

model_choice = st.sidebar.radio("Select Model", ['Linear Regression', 'Random Forest'])

# ----------------------
# Model Training
# ----------------------
if selected_features:
    X = df[selected_features]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_choice == 'Linear Regression':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader("📊 Model Performance")
    st.metric("RMSE", f"${rmse:.2f}")
    st.metric("R² Score", f"{r2:.2f}")

    if model_choice == 'Random Forest':
        st.subheader("🔍 Feature Importance")
        importances = pd.Series(model.feature_importances_, index=selected_features).sort_values()
        fig = px.bar(importances, orientation='h', labels={'value': 'Importance', 'index': 'Feature'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader("📈 Actual vs Predicted")
        comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        fig = px.scatter(comparison_df, x='Actual', y='Predicted', trendline='ols', title="Predicted vs Actual Prices")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please select at least one feature.")
