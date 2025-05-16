
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import datetime

# ----------------------
# PAGE SETUP
# ----------------------
st.set_page_config(layout="wide")
st.title("🛫 Flight Price Explorer: Revenue Steering Analysis")

# ----------------------
# LOAD & CLEAN DATA
# ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("all_flights.csv")
    df['departureTime'] = pd.to_datetime(df['departureTime'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['durationTime'] = pd.to_numeric(df['durationTime'], errors='coerce')
    df['weekday'] = df['departureTime'].dt.day_name()
    df['day_of_week'] = df['departureTime'].dt.weekday
    df['hour'] = df['departureTime'].dt.hour
    df['month'] = df['departureTime'].dt.month
    if 'wifi' not in df.columns:
        df['wifi'] = 'Unknown'

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

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ----------------------
# COLOR PALETTE
# ----------------------
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
custom_colors = ['#d71920', '#00235f', '#f9ba00', '#660000', '#800080', '#3366ff',
                 '#c3f550', '#fbaa3f', '#000000']

# ----------------------
# SIDEBAR FILTERS
# ----------------------
st.sidebar.header("Filters")
direct_airlines = ['SWISS', 'United', 'Delta']
lufthansa_group = ['Austrian', 'Brussels Airlines', 'Discover Airlines', 'Eurowings',
                   'Edelweiss Air', 'ITA', 'Air Dolomiti', 'Lufthansa', 'SWISS']
star_alliance = ['Aegean', 'Air Canada', 'Air China', 'Air India', 'Air New Zealand', 'ANA',
                 'Asiana Airlines', 'Austrian', 'Avianca', 'Brussels Airlines', 'CopaAirlines',
                 'Croatia Airlines', 'Egyptair', 'Ethiopian Airlines', 'Eva Air', 'LOT Polish Airlines',
                 'Lufthansa', 'Shenzhen Airlines', 'Singapore Airlines', 'South African Airways',
                 'SWISS', 'Tap Air Portugal', 'Thai', 'Turkish Airlines', 'United']
group_option = st.sidebar.radio("Airline Group", ['All Airlines', 'Direct Airlines', 'Lufthansa Group', 'Star Alliance'])

if group_option == 'Direct Airlines':
    airline_filter = direct_airlines
elif group_option == 'Lufthansa Group':
    airline_filter = lufthansa_group
elif group_option == 'Star Alliance':
    airline_filter = star_alliance
else:
    airline_filter = sorted(df['airline'].unique())

df_filtered = df[df['airline'].isin(airline_filter)]

min_price = int(df_filtered['price'].min())
max_price = int(df_filtered['price'].max())
price_range = st.sidebar.slider("Price Range ($)", min_value=min_price, max_value=max_price, value=(min_price, max_price))
df_filtered = df_filtered[(df_filtered['price'] >= price_range[0]) & (df_filtered['price'] <= price_range[1])]

# ----------------------
# PRICE ANALYSIS
# ----------------------
st.header("📊 Price Analysis")
col1, col2 = st.columns(2)

with col1:
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_day = df_filtered.groupby('weekday')['price'].mean().reindex(day_order).reset_index()
    fig = px.bar(df_day, x='weekday', y='price', title='Average Price by Day of Week',
                 labels={'price': 'Avg Price ($)', 'weekday': 'Day'}, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"💰 Cheapest day to fly: **{df_day.loc[df_day['price'].idxmin(), 'weekday']}**")

with col2:
    tod_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    df_tod = df_filtered.groupby('timeOfDay')['price'].mean().reindex(tod_order).reset_index()
    fig = px.bar(df_tod, x='timeOfDay', y='price', title='Average Price by Time of Day',
                 labels={'price': 'Avg Price ($)', 'timeOfDay': 'Time'}, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"💰 Cheapest time to fly: **{df_tod.loc[df_tod['price'].idxmin(), 'timeOfDay']}**")

st.subheader("Airline Price Comparison")
df_airline = df_filtered.groupby('airline')['price'].mean().reset_index()
fig = px.bar(df_airline, x='airline', y='price', color='airline',
             color_discrete_map=airline_colors,
             title='Average Price by Airline',
             labels={'price': 'Average Price ($)'}, text_auto=True)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)
