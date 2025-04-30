import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Custom color palette
customColors = ['#d71920', '#00235f', '#f9ba00', '#660000', '#000000', '#3366ff']

# Load data
st.set_page_config(layout="wide")
df = pd.read_csv("all_flights.csv")

# Clean and preprocess
nycAirports = ["JFK", "EWR", "LGA"]
swissAirports = ["ZRH", "GVA", "BSL"]

df['departureTime'] = pd.to_datetime(df['departureTime'], errors='coerce')
df['arrivalAirportTime'] = pd.to_datetime(df['arrivalAirportTime'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['durationMinutes'] = pd.to_numeric(df['durationTime'], errors='coerce')
df['carbonEmissionsThisFlight'] = pd.to_numeric(df.get('carbonEmissionsThisFlight'), errors='coerce')

# Define airlines to include
includedAirlines = ['SWISS', 'Delta', 'United', 'Lufthansa']
df = df[df['airline'].isin(includedAirlines)].copy()

# Airline colors
airline_colors = {
    'Lufthansa': '#FFD700',
    'SWISS': '#d71920',
    'Delta': '#00235f',
    'United': '#1a75ff',
}

# Derived columns
df['pricePerMinute'] = df['price'] / df['durationMinutes']
df['carbonDifferencePercent'] = ((df['carbonEmissionsThisFlight'] - df['carbonEmissionsThisFlight'].mean()) /
                                 df['carbonEmissionsThisFlight'].mean()) * 100

# Label flights
def classify_flight_type(row):
    if row['departureAirportID'] in nycAirports and row['arrivalAirportID'] in swissAirports:
        return 'Direct'
    return 'Connecting'

df['flightType'] = df.apply(classify_flight_type, axis=1)
df['legroom'] = df['legroom'].fillna("Extra reclining seat")
if "recliningAndLegroom" in df.columns:
    df.drop(columns=["recliningAndLegroom"], inplace=True)

directFlights = df[df['flightType'] == 'Direct'].copy()
connectingFlights = df[df['flightType'] == 'Connecting'].copy()

st.title("Flights from NYC to CH")

# Heatmap helper function
def plotHeatmap(df, valueCol, title, xaxisTitle, colorscale='Blues', width=800, height=500):
    df_clean = df[[valueCol, 'airline']].dropna()
    binned_col = pd.cut(df_clean[valueCol], bins=10)
    pivot = df_clean.groupby(['airline', binned_col]).size().unstack(fill_value=0)
    pivot['Total'] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False).drop(columns="Total")

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(interval) for interval in pivot.columns],
        y=pivot.index,
        colorscale=colorscale,
        colorbar=dict(title='Number of Flights')
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title=xaxisTitle,
        yaxis_title='Airline',
        template='plotly_white',
        width=width,
        height=height
    )

    st.plotly_chart(fig, use_container_width=True)

# Insert heatmaps
st.subheader("Heatmap: Carbon Difference Percent by Airline")
plotHeatmap(
    directFlights,
    valueCol='carbonDifferencePercent',
    title='Carbon Difference Percent by Airline',
    xaxisTitle='Carbon Difference Percent',
    colorscale='Reds'
)

st.subheader("Heatmap: Price by Airline")
plotHeatmap(
    directFlights,
    valueCol='price',
    title='Price by Airline',
    xaxisTitle='Price (USD)',
    colorscale='Reds'
)

st.subheader("Heatmap: Duration Time by Airline")
plotHeatmap(
    directFlights,
    valueCol='durationTime',
    title='Duration Time by Airline',
    xaxisTitle='Duration (min)',
    colorscale='Reds'
)
