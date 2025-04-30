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

# Derived column
df['pricePerMinute'] = df['price'] / df['durationMinutes']

# Label flights as Direct or Connecting
def classify_flight_type(row):
    if row['departureAirportID'] in nycAirports and row['arrivalAirportID'] in swissAirports:
        return 'Direct'
    return 'Connecting'

df['flightType'] = df.apply(classify_flight_type, axis=1)

# Clean legroom and drop unused columns
df['legroom'] = df['legroom'].fillna("Extra reclining seat")
if "recliningAndLegroom" in df.columns:
    df.drop(columns=["recliningAndLegroom"], inplace=True)

# Split into direct and connecting flights
directFlights = df[df['flightType'] == 'Direct'].copy()
connectingFlights = df[df['flightType'] == 'Connecting'].copy()

st.title("Flights from NYC to CH")

# Use all data for toggleable airline plots
filtered = df.sort_values('departureTime')

# Define airline colors
airline_colors = {
    'Delta': 'navy',
    'SWISS': 'red',
    'United': 'lightblue'
}

# Helper to create traces
def create_traces(df):
    traces = []
    for airline in df['airline'].unique():
        data = df[df['airline'] == airline]
        traces.append(go.Scatter(
            x=data['departureTime'],
            y=data['price'],
            mode='markers+lines',
            name=airline,
            hovertext=data['flightNumber'],
            marker=dict(color=airline_colors.get(airline, 'gray'))
        ))
    return traces

# Create traces for both flight types
direct_traces = create_traces(directFlights)
connecting_traces = create_traces(connectingFlights)

# Build figure
fig = go.Figure(data=direct_traces + connecting_traces)

# Toggle menu
fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=[
                dict(label="Direct Flights",
                     method="update",
                     args=[{"visible": [True]*len(direct_traces) + [False]*len(connecting_traces)},
                           {"title": "Price Over Time (Direct Flights)"}]),
                dict(label="Connecting Flights",
                     method="update",
                     args=[{"visible": [False]*len(direct_traces) + [True]*len(connecting_traces)},
                           {"title": "Price Over Time (Connecting Flights)"}])
            ],
            direction="down",
            showactive=True,
            x=0.5,
            xanchor="center",
            y=1.1,
            yanchor="top"
        )
    ],
    xaxis_title="Departure Time",
    yaxis_title="Price (USD)",
    legend_title_text="Toggle Airlines",
    hovermode="closest",
    height=600,
    legend=dict(
        title_font=dict(size=12),
        font=dict(size=11),
        orientation="v",
        x=1.02,
        y=1,
        xanchor='left',
        yanchor='top',
        bordercolor="LightGray",
        borderwidth=1,
        itemclick='toggle',
        itemdoubleclick='toggleothers'
    )
)

st.subheader("Price Over Time by Airline and Flight Type")
st.plotly_chart(fig, use_container_width=True)

# Carbon Emissions vs Price
st.subheader("Carbon Emissions vs Price by Airline")
carbon_fig = go.Figure()
for airline in df['airline'].unique():
    df_airline = df[df['airline'] == airline]
    carbon_fig.add_trace(go.Scatter(
        x=df_airline['carbonEmissionsThisFlight'],
        y=df_airline['price'],
        mode='markers',
        name=airline,
        hovertext=df_airline['flightNumber'],
        marker=dict(color=airline_colors.get(airline, 'gray'))
    ))

carbon_fig.update_layout(
    xaxis_title="Carbon Emissions (kg CO₂)",
    yaxis_title="Price (USD)",
    legend_title_text="Airline",
    hovermode="closest",
    height=600,
    legend=dict(
        itemclick="toggle",
        itemdoubleclick="toggleothers",
        x=1.02,
        y=1,
        bordercolor="LightGray",
        borderwidth=1
    )
)
st.plotly_chart(carbon_fig, use_container_width=True)

# Price Per Minute by Airline
st.subheader("Price Per Minute by Airline")
ppm_fig = go.Figure()
for airline in df['airline'].unique():
    df_airline = df[df['airline'] == airline].sort_values('departureTime')
    ppm_fig.add_trace(go.Scatter(
        x=df_airline['departureTime'],
        y=df_airline['pricePerMinute'],
        mode='lines+markers',
        name=airline,
        marker=dict(color=airline_colors.get(airline, 'gray'))
    ))

ppm_fig.update_layout(
    xaxis_title="Departure Time",
    yaxis_title="Price per Minute (USD)",
    legend_title_text="Airline",
    hovermode="closest",
    height=600,
    legend=dict(
        itemclick="toggle",
        itemdoubleclick="toggleothers",
        x=1.02,
        y=1,
        bordercolor="LightGray",
        borderwidth=1
    )
)
st.plotly_chart(ppm_fig, use_container_width=True)

# Histograms
col1, col2 = st.columns(2)
with col1:
    st.subheader("Histogram: Prices")
    fig4 = px.histogram(df, x='price')
    st.plotly_chart(fig4, use_container_width=True)

with col2:
    st.subheader("Histogram: Duration")
    fig5 = px.histogram(df, x='durationMinutes')
    st.plotly_chart(fig5, use_container_width=True)

# Airplane Types
st.subheader("Airplane Types by Airline")
airplane_count = directFlights.groupby(['airline', 'airplane']).size().unstack(fill_value=0)
airplane_count = airplane_count.loc[airplane_count.sum(axis=1).sort_values(ascending=False).index]
fig6 = go.Figure()
for i, col in enumerate(airplane_count.columns):
    fig6.add_trace(go.Bar(x=airplane_count.index, y=airplane_count[col], name=col,
                          marker_color=customColors[i % len(customColors)]))
fig6.update_layout(barmode='stack', title="Airplane Type by Airline", plot_bgcolor='white')
st.plotly_chart(fig6, use_container_width=True)

# Legroom
st.subheader("Legroom by Airline")
legroom_count = directFlights.groupby(['airline', 'legroom']).size().unstack(fill_value=0)
legroom_count = legroom_count.loc[legroom_count.sum(axis=1).sort_values(ascending=False).index]
fig7 = go.Figure()
for i, col in enumerate(legroom_count.columns):
    fig7.add_trace(go.Bar(x=legroom_count.index, y=legroom_count[col], name=col,
                          marker_color=customColors[i % len(customColors)]))
fig7.update_layout(barmode='stack', title="Legroom by Airline", plot_bgcolor='white')
st.plotly_chart(fig7, use_container_width=True)

# WiFi
st.subheader("WiFi Availability by Airline")
wifi_count = directFlights.groupby(['airline', 'wifi']).size().unstack(fill_value=0)
wifi_count = wifi_count.loc[wifi_count.sum(axis=1).sort_values(ascending=False).index]
fig8 = go.Figure()
for i, col in enumerate(wifi_count.columns):
    fig8.add_trace(go.Bar(x=wifi_count.index, y=wifi_count[col], name=col,
                          marker_color=customColors[i % len(customColors)]))
fig8.update_layout(barmode='stack', title="WiFi by Airline", plot_bgcolor='white')
st.plotly_chart(fig8, use_container_width=True)
