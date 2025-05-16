import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ----------------------
# PAGE SETUP
# ----------------------
st.set_page_config(layout="wide")
st.title("🛫 Flight Price Explorer: Book Smarter")

# ----------------------
# LOAD & CLEAN DATA
# ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("all_flights.csv")
    df['departureTime'] = pd.to_datetime(df['departureTime'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['durationMinutes'] = pd.to_numeric(df['totalTripDuration'], errors='coerce')
    df['carbonEmissionsThisFlight'] = pd.to_numeric(df['carbonEmissionsThisFlight'], errors='coerce')

    df['weekday'] = df['departureTime'].dt.day_name()
    df['hour'] = df['departureTime'].dt.hour

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
    return df.dropna(subset=['price', 'airline'])

df = load_data()

# ----------------------
# CONSTANTS
# ----------------------
direct_airlines = ['SWISS', 'United', 'Delta']
lufthansa_group = ['Austrian', 'Brussels Airlines', 'Discover Airlines', 'Eurowings', 'Edelweiss Air', 'ITA', 'Air Dolomiti', 'Lufthansa', 'SWISS']
star_alliance = ['Aegean', 'Air Canada', 'Air China', 'Air India', 'Air New Zealand', 'ANA', 'Asiana Airlines', 'Austrian', 'Avianca', 'Brussels Airlines', 'CopaAirlines', 'Croatia Airlines', 'Egyptair', 'Ethiopian Airlines', 'Eva Air', 'LOT Polish Airlines', 'Lufthansa', 'Shenzhen Airlines', 'Singapore Airlines', 'South African Airways', 'SWISS', 'Tap Air Portugal', 'Thai', 'Turkish Airlines', 'United']

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
# SIDEBAR FILTERS
# ----------------------
st.sidebar.header("Filters")
group_option = st.sidebar.radio("Airline Group", ['All Airlines', 'Direct Airlines', 'Lufthansa Group', 'Star Alliance'])

if group_option == 'Direct Airlines':
    df = df[df['airline'].isin(direct_airlines)]
elif group_option == 'Lufthansa Group':
    df = df[df['airline'].isin(lufthansa_group)]
elif group_option == 'Star Alliance':
    df = df[df['airline'].isin(star_alliance)]

# ----------------------
# PRICE CHARTS
# ----------------------
st.subheader("🎯 Average Price by Key Time Features")

time_chart_type = st.selectbox("Group price by:", ['Day of Week', 'Time of Day'])

if time_chart_type == 'Day of Week':
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_grouped = df.groupby(['weekday', 'airline'])['price'].mean().reset_index()
    fig = px.bar(
        df_grouped,
        x='weekday', y='price', color='airline',
        category_orders={'weekday': day_order},
        labels={'price': 'Avg Price ($)'},
        title='Average Price by Day of Week',
        color_discrete_map=airline_colors,
        barmode='group'
    )
else:
    tod_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    df_grouped = df.groupby(['timeOfDay', 'airline'])['price'].mean().reset_index()
    fig = px.bar(
        df_grouped,
        x='timeOfDay', y='price', color='airline',
        category_orders={'timeOfDay': tod_order},
        labels={'price': 'Avg Price ($)'},
        title='Average Price by Time of Day',
        color_discrete_map=airline_colors,
        barmode='group'
    )

st.plotly_chart(fig, use_container_width=True)

# ----------------------
# MODELING
# ----------------------
st.subheader("📈 Price Prediction Model")
st.markdown("Predict price based on departure hour and weekday")

model_data = df[['price', 'hour']].copy()
model_data['weekday_num'] = df['departureTime'].dt.weekday

X = model_data[['hour', 'weekday_num']]
y = model_data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

st.write(f"Model RMSE: ${rmse:.2f}")
st.success(f"📌 Cheapest predicted time to book: {df.groupby('hour')['price'].mean().idxmin()}:00")
