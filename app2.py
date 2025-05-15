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

custom_colors = ['#d71920', '#00235f', '#f9ba00', '#660000', '#800080', '#3366ff',
                '#c3f550', '#fbaa3f', '#000000']

airline_colors = {
    'Lufthansa': '#ffd700',
    'SWISS': '#d71920',
    'Delta': '#00235f',
    'United': '#1a75ff',
    'Edelweiss Air': '#800080',
    'Air Dolomiti': '#32cd32',
    'Austrian': '#c3f550',
    'ITA': '#fbaa3f',
    'Brussels Airlines': '#00235f',
    'Eurowings': '#1a75ff',
    'Aegean': '#767676',
    'Air Canada': '#00235f',
    'Tap Air Portugal': '#fbaa3f',
    'Turkish Airlines': '#800080'
}

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
# MULTIPLE PREDICTION MODELS
# --------------------------
st.subheader("🤖 Predictive Modeling: When to Buy")
st.markdown("""
These models predict **average flight price** based on the time of day (hour) and month of travel.
This helps determine **when is the most affordable time to book flights**.
""")
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
# CARBON EMISSIONS
# --------------------------
st.subheader("🌍 Carbon Emissions Overview")
df['aircraftType'] = df['aircraft'].apply(classify_aircraft)
emissions_by_aircraft = df.groupby(['aircraftType', 'airline'])['carbonEmissionsThisFlight'].mean().reset_index()
fig2 = px.bar(emissions_by_aircraft, x='aircraftType', y='carbonEmissionsThisFlight', color='airline', title="Average CO₂ Emissions by Aircraft Type and Airline", labels={"carbonEmissionsThisFlight": "Avg CO₂ (kg)", "aircraftType": "Aircraft Type"}, barmode='group', color_discrete_map=airline_colors)
st.plotly_chart(fig2, use_container_width=True)

# --------------------------
# ROUTE EFFICIENCY
# --------------------------
st.subheader("⛽ Route Efficiency Analytics")
df['efficiency'] = df['durationMinutes'] / df['carbonEmissionsThisFlight']
origin_col = 'departureAirportID'
destination_col = 'arrivalAirportID'

if origin_col in df.columns and destination_col in df.columns:
    efficiency_by_route = (
        df.groupby([origin_col, destination_col])['efficiency']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    st.dataframe(
        efficiency_by_route.head(10).rename(columns={
            origin_col: 'From',
            destination_col: 'To',
            'efficiency': 'Minutes per kg CO₂'
        }),
        use_container_width=True
    )
else:
    st.warning("⚠️ Could not find columns for origin and destination airports. Please check your CSV.")

# --------------------------
# SUSTAINABILITY SCORE
# --------------------------
st.subheader("♻️ Sustainability-Focused Insights")
df['sustainabilityScore'] = 100 - (df['carbonEmissionsThisFlight'] / df['durationMinutes']) * 10
score_df = df.groupby('airline')['sustainabilityScore'].mean().sort_values(ascending=False).reset_index()
fig3 = px.bar(score_df, x='airline', y='sustainabilityScore', title="Sustainability Score by Airline", labels={"sustainabilityScore": "Score", "airline": "Airline"}, color='airline', color_discrete_map=airline_colors)
st.plotly_chart(fig3, use_container_width=True)

# --------------------------
# TIME-BASED PRICE DISTRIBUTION
# --------------------------
st.subheader("🕒 Price Distribution by Time of Day and Weekday")
st.markdown("**Time of day breakdown:** Morning (5–12), Afternoon (12–17), Evening (17–22), Night (22–5)")

price_by_day = df.groupby(['weekday', 'airline'])['price'].mean().reset_index()
price_by_time = df.groupby(['timeOfDay', 'airline'])['price'].mean().reset_index()

fig_day = px.bar(price_by_day, x='weekday', y='price', color='airline', title="Average Price by Day of Week and Airline", labels={"price": "Avg Price", "weekday": "Day of Week"}, barmode='group', category_orders={"weekday": ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}, color_discrete_map=airline_colors)
fig_time = px.bar(price_by_time, x='timeOfDay', y='price', color='airline', title="Average Price by Time of Day and Airline", labels={"price": "Avg Price", "timeOfDay": "Time of Day"}, barmode='group', category_orders={"timeOfDay": ['Morning', 'Afternoon', 'Evening', 'Night']}, color_discrete_map=airline_colors)

st.plotly_chart(fig_day, use_container_width=True)
st.plotly_chart(fig_time, use_container_width=True)
