import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ----------------------------
# PAGE CONFIG & STYLE
# ----------------------------
st.set_page_config(layout="wide", page_title="Flight Pricing Intelligence", page_icon="✈️")
st.title("✈️ Flight Price & Sustainability Insights")

st.markdown("""
<style>
    .css-18e3th9 { padding-top: 1rem; padding-bottom: 1rem; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# AIRLINE COLORS
# ----------------------------
airline_colors = {
    'SWISS': '#d71920',
    'Delta': '#00235f',
    'United': '#1a75ff',
    'Lufthansa': '#ffd700',
    'Edelweiss Air': '#800080',
    'Air Dolomiti': '#32cd32',
    'Austrian': '#c3f550',
    'ITA': '#fbaa3f',
    'Brussels Airlines': '#88c0d0',
    'Eurowings': '#a3be8c',
    'Aegean': '#b48ead',
    'Air Canada': '#5e81ac',
    'Tap Air Portugal': '#ebcb8b',
    'Turkish Airlines': '#d08770'
}

# ----------------------------
# AIRLINE GROUPS
# ----------------------------
direct_airlines = ['SWISS', 'United', 'Delta']
lufthansa_group = ['Austrian', 'Brussels Airlines', 'Discover Airlines', 'Eurowings', 'Edelweiss Air', 'ITA', 'Air Dolomiti', 'Lufthansa', 'SWISS']
star_alliance = ['Aegean', 'Air Canada', 'Air China', 'Air India', 'Air New Zealand', 'ANA', 'Asiana Airlines', 'Austrian', 'Avianca',
                 'Brussels Airlines', 'CopaAirlines', 'Croatia Airlines', 'Egyptair', 'Ethiopian Airlines', 'Eva Air', 'LOT Polish Airlines',
                 'Lufthansa', 'Shenzhen Airlines', 'Singapore Airlines', 'South African Airways', 'SWISS', 'Tap Air Portugal', 'Thai',
                 'Turkish Airlines', 'United']

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def classify_aircraft(aircraft):
    if pd.isna(aircraft): return "Other"
    aircraft = str(aircraft).lower()
    if aircraft.startswith("airbus"): return "Airbus"
    elif aircraft.startswith("boeing"): return "Boeing"
    elif aircraft.startswith("canadair"): return "Canadair"
    elif aircraft.startswith("embraer"): return "Embraer"
    return "Other"

def time_of_day_label(hour):
    if 5 <= hour < 12: return 'Morning (5–12)'
    elif 12 <= hour < 17: return 'Afternoon (12–17)'
    elif 17 <= hour < 22: return 'Evening (17–22)'
    return 'Night (22–5)'

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("all_flights.csv")
    df['departureTime'] = pd.to_datetime(df['departureTime'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['durationMinutes'] = pd.to_numeric(df['durationTime'], errors='coerce')
    df['carbonEmissionsThisFlight'] = pd.to_numeric(df.get('carbonEmissionsThisFlight'), errors='coerce')
    df['aircraft'] = df['airplane'].fillna('Unknown')
    df['weekday'] = df['departureTime'].dt.day_name()
    df['hour'] = df['departureTime'].dt.hour
    df['month'] = df['departureTime'].dt.month
    df['date'] = df['departureTime'].dt.date
    df['timeOfDay'] = df['hour'].apply(time_of_day_label)
    return df.dropna(subset=['price', 'durationMinutes', 'carbonEmissionsThisFlight'])

df = load_data()

# ----------------------------
# SIDEBAR FILTER
# ----------------------------
st.sidebar.header("🎛️ Filters")
group_option = st.sidebar.radio("Select Airline Group", ['Direct Airlines', 'Lufthansa Group', 'Star Alliance'])
if group_option == 'Direct Airlines':
    selected_airlines = direct_airlines
elif group_option == 'Lufthansa Group':
    selected_airlines = lufthansa_group
else:
    selected_airlines = star_alliance

df = df[df['airline'].isin(selected_airlines)]

# ----------------------------
# PRICE TRENDS
# ----------------------------
st.subheader("📈 Historical Price Trends")
price_by_date = df.groupby(['date', 'airline'])['price'].mean().reset_index()
fig1 = px.line(price_by_date, x='date', y='price', color='airline', color_discrete_map=airline_colors,
               title="Average Ticket Price Over Time", labels={'price': 'Price ($)', 'date': 'Departure Date'})
st.plotly_chart(fig1, use_container_width=True)

# ----------------------------
# PREDICTIVE MODELING
# ----------------------------
st.subheader("🤖 Predictive Modeling: When to Book")
st.markdown("These models predict flight prices based on **hour of day** and **month**, helping you identify optimal booking windows.")

X = df[['hour', 'month']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    st.markdown(f"**{name} RMSE**: ${rmse:.2f}")

best_hour = df.groupby('hour')['price'].mean().idxmin()
best_month = df.groupby('month')['price'].mean().idxmin()
st.success(f"🕒 Best Hour to Book: {best_hour}:00 📅 Best Month: {best_month}")

# ----------------------------
# CO2 EMISSIONS BY AIRCRAFT
# ----------------------------
st.subheader("🌍 Carbon Emissions by Aircraft Type and Airline")
df['aircraftType'] = df['aircraft'].apply(classify_aircraft)
emissions_by_aircraft = df.groupby(['aircraftType', 'airline'])['carbonEmissionsThisFlight'].mean().reset_index()
fig2 = px.bar(emissions_by_aircraft, x='aircraftType', y='carbonEmissionsThisFlight', color='airline',
              barmode='group', color_discrete_map=airline_colors,
              title="Average CO₂ Emissions by Aircraft Type", labels={'carbonEmissionsThisFlight': 'CO₂ (kg)', 'aircraftType': 'Aircraft Type'})
st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# SUSTAINABILITY SCORE
# ----------------------------
st.subheader("♻️ Sustainability Score by Airline")
df['sustainabilityScore'] = 100 - (df['carbonEmissionsThisFlight'] / df['durationMinutes']) * 10
score_df = df.groupby('airline')['sustainabilityScore'].mean().sort_values(ascending=False).reset_index()
fig3 = px.bar(score_df, x='airline', y='sustainabilityScore', color='airline',
              color_discrete_map=airline_colors,
              title="Sustainability Score (Lower CO₂ per Minute)", labels={'sustainabilityScore': 'Score'})
st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# TIME & WEEKDAY PRICE DISTRIBUTIONS
# ----------------------------
st.subheader("🕒 Price Distribution by Time of Day and Weekday")
st.markdown("**Time of day breakdown:** Morning (5–12), Afternoon (12–17), Evening (17–22), Night (22–5)")

weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
fig_day = px.bar(df.groupby(['weekday', 'airline'])['price'].mean().reset_index(),
                 x='weekday', y='price', color='airline',
                 barmode='group', category_orders={'weekday': weekday_order},
                 color_discrete_map=airline_colors,
                 title="Average Price by Day of Week and Airline",
                 labels={'price': 'Avg Price ($)', 'weekday': 'Day of Week'})
st.plotly_chart(fig_day, use_container_width=True)

timeofday_order = ['Morning (5–12)', 'Afternoon (12–17)', 'Evening (17–22)', 'Night (22–5)']
fig_time = px.bar(df.groupby(['timeOfDay', 'airline'])['price'].mean().reset_index(),
                  x='timeOfDay', y='price', color='airline',
                  barmode='group', category_orders={'timeOfDay': timeofday_order},
                  color_discrete_map=airline_colors,
                  title="Average Price by Time of Day and Airline",
                  labels={'price': 'Avg Price ($)', 'timeOfDay': 'Time of Day'})
st.plotly_chart(fig_time, use_container_width=True)
