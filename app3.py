import streamlit as st
import pandas as pd
import plotly.express as px

# ----------------------
# PAGE SETUP
# ----------------------
st.set_page_config(layout="wide")
st.title("Flight Price Exploration (NYC to CH): Revenue Steering Analysis")

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
    df['dayOfWeek'] = df['departureTime'].dt.weekday
    df['hour'] = df['departureTime'].dt.hour
    df['month'] = df['departureTime'].dt.month
    df['airline'] = df['airline'].astype(str).str.strip()
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

# Load and filter data
df = load_data()

# ROUTE FILTERING: NYC to SWITZERLAND
nyc_airports = ["LGA", "JFK", "EWR"]
swiss_airports = ["ZRH", "BSL", "GVA"]

if 'departureAirportID' in df.columns and 'arrivalAirportID' in df.columns:
    df = df[df['departureAirportID'].isin(nyc_airports) & df['arrivalAirportID'].isin(swiss_airports)]

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
# Define airline groups
direct_airlines = ['SWISS', 'United', 'Delta']
lufthansa_group = ['Austrian', 'Brussels Airlines', 'Discover Airlines', 'Eurowings', 'Edelweiss Air', 'ITA', 'Air Dolomiti', 'Lufthansa', 'SWISS']
star_alliance = ['Aegean', 'Air Canada', 'Air China', 'Air India', 'Air New Zealand', 'ANA', 'Asiana Airlines', 'Austrian', 'Avianca', 'Brussels Airlines', 'CopaAirlines', 'Croatia Airlines', 'Egyptair', 'Ethiopian Airlines', 'Eva Air', 'LOT Polish Airlines', 'Lufthansa', 'Shenzhen Airlines', 'Singapore Airlines', 'South African Airways', 'SWISS', 'Tap Air Portugal', 'Thai', 'Turkish Airlines', 'United']

# Sidebar filters
st.sidebar.title("Filter Flights")
filter_option = st.sidebar.radio(
    "Choose flight group:",
    ("All Flights", "Direct Flights", "Lufthansa Group", "Star Alliance")
)

if filter_option == "Direct Flights":
    df_filtered = df[df["airline"].isin(direct_airlines)]
elif filter_option == "Lufthansa Group":
    df_filtered = df[df["airline"].isin(lufthansa_group)]
elif filter_option == "Star Alliance":
    df_filtered = df[df["airline"].isin(star_alliance)]
else:
    df_filtered = df.copy()

# Header
st.title("Flight Explorer")

# Summary stats
st.subheader("Summary Statistics")
st.metric("Average Price ($)", int(df_filtered["price"].mean()))
st.metric("Average Duration (mins)", int(df_filtered["durationTime"].mean()))
st.metric("Avg Carbon Emissions", int(df_filtered["carbonEmissionsThisFlight"].mean()))

# Charts
st.subheader("Price Distribution")
fig1 = px.histogram(df_filtered, x="price", nbins=30, title="Flight Price Distribution")
st.plotly_chart(fig1)

st.subheader("Duration vs Price")
fig2 = px.scatter(df_filtered, x="durationTime", y="price", color="airline", title="Duration vs Price")
st.plotly_chart(fig2)

st.subheader("Carbon Emissions by Airline")
carbon_grouped = df_filtered.groupby("airline")["carbonEmissionsThisFlight"].mean().reset_index()
fig3 = px.bar(carbon_grouped.sort_values(by="carbonEmissionsThisFlight"), x="airline", y="carbonEmissionsThisFlight", title="Avg Carbon Emissions by Airline")
st.plotly_chart(fig3)
