import streamlit as st
import pandas as pd
import plotly.express as px

# ----------------------
# LOAD DATA
# ----------------------
df = pd.read_csv("all_flights.csv")

# ----------------------
# AIRLINE GROUPS
# ----------------------
direct_airlines = ['SWISS', 'United', 'Delta']
lufthansa_group = ['Austrian', 'Brussels Airlines', 'Discover Airlines', 'Eurowings', 'Edelweiss Air', 'ITA', 'Air Dolomiti', 'Lufthansa', 'SWISS']
star_alliance = ['Aegean', 'Air Canada', 'Air China', 'Air India', 'Air New Zealand', 'ANA', 'Asiana Airlines', 'Austrian', 'Avianca', 'Brussels Airlines', 'CopaAirlines', 'Croatia Airlines', 'Egyptair', 'Ethiopian Airlines', 'Eva Air', 'LOT Polish Airlines', 'Lufthansa', 'Shenzhen Airlines', 'Singapore Airlines', 'South African Airways', 'SWISS', 'Tap Air Portugal', 'Thai', 'Turkish Airlines', 'United']

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
custom_colors = ['#d71920', '#00235f', '#f9ba00', '#660000', '#800080', '#3366ff', '#c3f550', '#fbaa3f', '#000000']

# ----------------------
# TIME FEATURE ENGINEERING
# ----------------------
df['timestamp'] = pd.to_datetime(df['departureDateTime'], errors='coerce')
df['dayOfWeek'] = df['timestamp'].dt.day_name()
df['hour'] = df['timestamp'].dt.hour

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

# ----------------------
# ROUTE FILTERING
# ----------------------
nyc_airports = ["LGA", "JFK", "EWR"]
swiss_airports = ["ZRH", "BSL", "GVA"]
df = df[df['departureAirportID'].isin(nyc_airports) & df['arrivalAirportID'].isin(swiss_airports)]

# ----------------------
# FILTER SIDEBAR
# ----------------------
st.set_page_config(layout="wide")
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

# ----------------------
# MAIN PAGE
# ----------------------
st.title("Flight Price Exploration (NYC to CH): Revenue Steering Analysis")

# Summary
st.subheader("Summary Statistics")
st.metric("Average Price ($)", int(df_filtered["price"].mean()))
st.metric("Average Duration (mins)", int(df_filtered["durationTime"].mean()))
st.metric("Avg Carbon Emissions", int(df_filtered["carbonEmissionsThisFlight"].mean()))

# Price distribution
st.subheader("Price Distribution")
fig1 = px.histogram(df_filtered, x="price", nbins=30, title="Flight Price Distribution")
st.plotly_chart(fig1)

# Duration vs Price
st.subheader("Duration vs Price")
fig2 = px.scatter(df_filtered, x="durationTime", y="price", color="airline", title="Duration vs Price",
                  color_discrete_map=airline_colors)
st.plotly_chart(fig2)

# Carbon Emissions
st.subheader("Carbon Emissions by Airline")
carbon_grouped = df_filtered.groupby("airline")["carbonEmissionsThisFlight"].mean().reset_index()
fig3 = px.bar(carbon_grouped.sort_values(by="carbonEmissionsThisFlight"),
              x="airline", y="carbonEmissionsThisFlight", title="Avg Carbon Emissions by Airline",
              color="airline", color_discrete_map=airline_colors)
st.plotly_chart(fig3)

# Additional EDA (Time Features)
st.subheader("Flight Count by Day of Week")
fig4 = px.histogram(df_filtered, x="dayOfWeek", color="airline", title="Flights by Day of Week",
                    color_discrete_map=airline_colors, category_orders={"dayOfWeek": ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']})
st.plotly_chart(fig4)

st.subheader("Flight Count by Time of Day")
fig5 = px.histogram(df_filtered, x="timeOfDay", color="airline", title="Flights by Time of Day",
                    color_discrete_map=airline_colors, category_orders={"timeOfDay": ['Morning', 'Afternoon', 'Evening', 'Night']})
st.plotly_chart(fig5)
