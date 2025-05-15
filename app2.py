import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

# Page setup
st.set_page_config(layout="wide")
st.title("✈️ Flight Price & Sustainability Insights")

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("all_flights.csv")

    # Clean and engineer features
    df['departureTime'] = pd.to_datetime(df['departureTime'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['durationMinutes'] = pd.to_numeric(df['durationTime'], errors='coerce')
    df['carbonEmissionsThisFlight'] = pd.to_numeric(df.get('carbonEmissionsThisFlight'), errors='coerce')
    df['carbonEmissionsTypicalRoute'] = pd.to_numeric(df.get('carbonEmissionsTypicalRoute'), errors='coerce')
    df['aircraft'] = df['airplane'].fillna('Unknown')
    df['weekday'] = df['departureTime'].dt.day_name()
    df['hour'] = df['departureTime'].dt.hour
    df['month'] = df['departureTime'].dt.month
    df['date'] = df['departureTime'].dt.date

    return df.dropna(subset=['price', 'durationMinutes', 'carbonEmissionsThisFlight'])

df = load_data()

# Sidebar filter
st.sidebar.header("Filters")
airlines = df['airline'].dropna().unique()
selected_airlines = st.sidebar.multiselect("Airlines", airlines, default=airlines)
df = df[df['airline'].isin(selected_airlines)]

# --------------------------
# 1. Flight Price Trends
# --------------------------
st.subheader("📈 Historical Price Trends")
price_by_date = df.groupby('date')['price'].mean().reset_index()
fig1 = px.line(price_by_date, x='date', y='price', title="Average Ticket Price Over Time")
st.plotly_chart(fig1, use_container_width=True)

# --------------------------
# 2. Predict Best Time to Buy
# --------------------------
st.subheader("🤖 Predictive Modeling: When to Buy")
model_df = df[['price', 'hour', 'month']]
X = model_df[['hour', 'month']]
y = model_df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.markdown(f"**Model RMSE**: ${rmse:.2f}")
best_hour = int(df.groupby('hour')['price'].mean().idxmin())
best_month = int(df.groupby('month')['price'].mean().idxmin())
st.success(f"📌 Best time to book: **Hour {best_hour}:00**, Month {best_month}")

# --------------------------
# 3. Carbon Emissions Analysis
# --------------------------
st.subheader("🌍 Carbon Emissions Overview")

emissions_by_aircraft = df.groupby('aircraft')['carbonEmissionsThisFlight'].mean().sort_values()
fig2 = px.bar(emissions_by_aircraft, title="Average CO₂ Emissions by Aircraft Type")
st.plotly_chart(fig2, use_container_width=True)

# --------------------------
# 4. Fuel Efficiency by Route
# --------------------------
st.subheader("⛽ Route Efficiency Analytics")

df['efficiency'] = df['durationMinutes'] / df['carbonEmissionsThisFlight']
efficiency_by_route = df.groupby(['originAirport', 'destinationAirport'])['efficiency'].mean().sort_values(ascending=False).reset_index()

st.dataframe(efficiency_by_route.head(10).rename(columns={
    'originAirport': 'From',
    'destinationAirport': 'To',
    'efficiency': 'Minutes per kg CO₂'
}), use_container_width=True)

# --------------------------
# 5. Sustainability Scoring
# --------------------------
st.subheader("♻️ Sustainability-Focused Insights")

df['sustainabilityScore'] = 100 - (df['carbonEmissionsThisFlight'] / df['durationMinutes']) * 10
score_df = df.groupby('airline')['sustainabilityScore'].mean().sort_values(ascending=False)
fig3 = px.bar(score_df, title="Sustainability Score by Airline")
st.plotly_chart(fig3, use_container_width=True)
