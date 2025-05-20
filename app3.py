import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv("all_flights.csv")

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
