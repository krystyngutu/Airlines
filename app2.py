import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

# ----------------------
# PAGE SETUP
# ----------------------
st.set_page_config(layout="wide")
st.title("✈️ Flight Price & Sustainability Insights")

# ----------------------
# CONSTANTS
# ----------------------
direct_airlines = ['SWISS', 'United', 'Delta']
lufthansa_group = ['Austrian', 'Brussels Airlines', 'Discover Airlines', 'Eurowings', 'Edelweiss Air', 'ITA', 'Air Dolomiti', 'Lufthansa', 'SWISS']
star_alliance = ['Aegean', 'Air Canada', 'Air China', 'Air India', 'Air New Zealand', 'ANA', 'Asiana Airlines', 'Austrian', 'Avianca', 'Brussels Airlines', 'CopaAirlines', 'Croatia Airlines', 'Egyptair', 'Ethiopian Airlines', 'Eva Air', 'LOT Polish Airlines', 'Lufthansa', 'Shenzhen Airlines', 'Singapore Airlines', 'South African Airways', 'SWISS', 'Tap Air Portugal', 'Thai', 'Turkish Airlines', 'United']

custom_colors = ['#88d498', '#f4e285', '#f4a259', '#8cb369', '#f2c14e', '#c8553d', '#3f612d', '#a1c349', '#f6ae2d']

airline_colors = {airline: color for airline, color in zip(star_alliance, custom_colors * 2)}

# ----------------------
# HELPER FUNCTIONS
# ----------------------
def classify_aircraft(aircraft):
    if pd.isna(aircraft):
        return "Other"
    aircraft = str(aircraft).lower()
    if aircraft.startswith("airbus"):
        return "Airbus"
    elif aircraft.startswith("boeing"):
        return "Boeing"
    elif aircraft.startswith("canadair"):
        return "Canadair"
    elif aircraft.startswith("embraer"):
        return "Embraer"
    else:
        return "Other"

# ----------------------
# LOAD DATA
# ----------------------
@st.cache_data

def load_data():
    df = pd.read_csv("all_flights.csv")
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

    def time_of_day(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'

    df['timeOfDay'] = df['hour'].apply(time_of_day)
    df['timeOfDayLabel'] = df['timeOfDay'].map({
        'Morning': 'Morning (5–12)',
        'Afternoon': 'Afternoon (12–17)',
        'Evening': 'Evening (17–22)',
        'Night': 'Night (22–5)'
    })
    return df.dropna(subset=['price', 'durationMinutes', 'carbonEmissionsThisFlight'])

df = load_data()

# ----------------------
# SIDEBAR FILTER
# ----------------------
st.sidebar.header("Filters")
group_option = st.sidebar.radio("Airline Group", ['Direct Airlines', 'Lufthansa Group', 'Star Alliance'])

if group_option == 'Direct Airlines':
    selected_airlines = direct_airlines
elif group_option == 'Lufthansa Group':
    selected_airlines = lufthansa_group
else:
    selected_airlines = star_alliance

df = df[df['airline'].isin(selected_airlines)]

# --------------------------
# PRICE TRENDS
# --------------------------
st.subheader("📈 Historical Price Trends")
price_by_date = df.groupby(['date', 'airline'])['price'].mean().reset_index()
fig1 = px.line(price_by_date, x='date', y='price', color='airline', title="Average Ticket Price Over Time", color_discrete_map=airline_colors)
st.plotly_chart(fig1, use_container_width=True)

# --------------------------
# PREDICTIVE MODELING
# --------------------------
st.subheader("🤖 Predictive Modeling: When to Buy")
st.markdown("These models predict **average flight price** based on time of day and month to help determine when to book.")

model_df = df[['price', 'hour', 'month']]
X = model_df[['hour', 'month']]
y = model_df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.markdown(f"**{name} RMSE**: ${rmse:.2f}")

best_hour = int(df.groupby('hour')['price'].mean().idxmin())
best_month = int(df.groupby('month')['price'].mean().idxmin())
st.success(f"📌 Best time to book: **Hour {best_hour}:00**, Month {best_month}")

# --------------------------
# TIME OF DAY CHART
# --------------------------
st.subheader("🕒 Avg Price by Time of Day")
st.markdown("**Morning (5–12)**, **Afternoon (12–17)**, **Evening (17–22)**, **Night (22–5)**")
price_by_time = df.groupby(['timeOfDayLabel', 'airline'])['price'].mean().reset_index()
fig_time = px.bar(
    price_by_time,
    x='timeOfDayLabel',
    y='price',
    color='airline',
    title='Average Price by Time of Day and Airline',
    labels={'price': 'Avg Price ($)', 'timeOfDayLabel': 'Time of Day'},
    barmode='group',
    color_discrete_map=airline_colors
)
st.plotly_chart(fig_time, use_container_width=True)
