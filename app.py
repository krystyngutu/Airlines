import pandas as pd
import streamlit as st
import plotly.express as px

# Load CSV file
st.set_page_config(layout='wide')
df = pd.read_csv("all_flights.csv")

# Clean and process
df['departureTime'] = pd.to_datetime(df['departureTime'], errors='coerce')
df['arrivalAirportTime'] = pd.to_datetime(df['arrivalAirportTime'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['durationMinutes'] = pd.to_numeric(df['durationTime'], errors='coerce')
df['carbonEmissionsThisFlight'] = pd.to_numeric(df.get('carbonEmissionsThisFlight'), errors='coerce')

# Define  airports and airlines to look at
nycAirports = ['JFK', 'EWR', 'LGA']
swissAirports = ['ZRH', 'GVA', 'BSL']

includedAirlines = ['SWISS', 'Lufthansa', 'Edelweiss Air', 'Delta', 'United']

# Filter to only include selected airlines
df = df[df['airline'].isin(includedAirlines)].copy()

# Dashboard Title
st.title("Flight Dashboard: NYC to CH")

# Price over time
st.subheader("Price Over Time")
fig = px.bar(filtered, x='departureTime', y='price', hover_data=['flightNumber'])
st.plotly_chart(fig)

# Carbon emissions vs price
st.subheader("Carbon Emissions vs Price")
fig = px.scatter(filtered, x='carbonEmissionsThisFlight', y='price', hover_data=['departureTime'])
st.plotly_chart(fig)

# Histogram of Prices
st.subheader("Histogram: Prices")
fig = px.histogram(filtered, x='price')
st.plotly_chart(fig)

# Histogram of Duration
st.subheader("Histogram: Duration")
fig = px.histogram(filtered, x='durationMinutes')
st.plotly_chart(fig)

# Show raw data
st.subheader("Raw Data")
st.dataframe(filtered)
