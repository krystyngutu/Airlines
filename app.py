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

# Define  airlines to include
includedAirlines = ['SWISS', 'Delta', 'United', 'Lufthansa', 'British Airways',
                    'Air Canada', 'Air France', 'KLM', 'American', 'Scandinavian Airlines']

# Filter to only include selected airlines
df = df[df['airline'].isin(includedAirlines)].copy()

# Define new airline colors
airline_colors = {
    'Lufthansa': '#FFD700',           # gold
    'SWISS': '#d71920',               # red
    'Delta': '#00235f',               # dark blue
    'United': '#1a75ff',              # light blue
    'British Airways': '#660000',     # dark red
    'Air Canada': '#000000',          # black
    'Air France': '#3366ff',          # royal blue
    'KLM': '#00BFFF',                 # sky blue
    'American': '#8B0000',            # deep red
    'Scandinavian Airlines': '#708090' # slate gray
}

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

# Build interactive figure
fig = go.Figure(data=direct_traces + connecting_traces)
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

# ---------- Carbon Emissions vs Price ----------
st.subheader("Carbon Emissions vs Price by Airline and Flight Type")

# Prepare traces
carbon_direct_traces = []
carbon_connecting_traces = []

for airline in df['airline'].unique():
    # Direct
    data_direct = directFlights[directFlights['airline'] == airline]
    carbon_direct_traces.append(go.Scatter(
        x=data_direct['carbonEmissionsThisFlight'],
        y=data_direct['price'],
        mode='markers',
        name=airline,
        hovertext=data_direct['flightNumber'],
        marker=dict(color=airline_colors.get(airline, 'gray'))
    ))

    # Connecting
    data_connecting = connectingFlights[connectingFlights['airline'] == airline]
    carbon_connecting_traces.append(go.Scatter(
        x=data_connecting['carbonEmissionsThisFlight'],
        y=data_connecting['price'],
        mode='markers',
        name=airline,
        hovertext=data_connecting['flightNumber'],
        marker=dict(color=airline_colors.get(airline, 'gray'))
    ))

# Combine into figure
carbon_fig = go.Figure(data=carbon_direct_traces + carbon_connecting_traces)

carbon_fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=[
                dict(label="Direct Flights",
                     method="update",
                     args=[{"visible": [True]*len(carbon_direct_traces) + [False]*len(carbon_connecting_traces)},
                           {"title": "Carbon Emissions vs Price (Direct Flights)"}]),
                dict(label="Connecting Flights",
                     method="update",
                     args=[{"visible": [False]*len(carbon_direct_traces) + [True]*len(carbon_connecting_traces)},
                           {"title": "Carbon Emissions vs Price (Connecting Flights)"}])
            ],
            direction="down",
            showactive=True,
            x=0.5,
            xanchor="center",
            y=1.1,
            yanchor="top"
        )
    ],
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

# ---------- Price Per Minute ----------
st.subheader("Price Per Minute by Airline and Flight Type")

# Prepare traces
ppm_direct_traces = []
ppm_connecting_traces = []

for airline in df['airline'].unique():
    # Direct
    data_direct = directFlights[directFlights['airline'] == airline].sort_values('departureTime')
    ppm_direct_traces.append(go.Scatter(
        x=data_direct['departureTime'],
        y=data_direct['pricePerMinute'],
        mode='lines+markers',
        name=airline,
        marker=dict(color=airline_colors.get(airline, 'gray'))
    ))

    # Connecting
    data_connecting = connectingFlights[connectingFlights['airline'] == airline].sort_values('departureTime')
    ppm_connecting_traces.append(go.Scatter(
        x=data_connecting['departureTime'],
        y=data_connecting['pricePerMinute'],
        mode='lines+markers',
        name=airline,
        marker=dict(color=airline_colors.get(airline, 'gray'))
    ))

# Combine into figure
ppm_fig = go.Figure(data=ppm_direct_traces + ppm_connecting_traces)

ppm_fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=[
                dict(label="Direct Flights",
                     method="update",
                     args=[{"visible": [True]*len(ppm_direct_traces) + [False]*len(ppm_connecting_traces)},
                           {"title": "Price Per Minute (Direct Flights)"}]),
                dict(label="Connecting Flights",
                     method="update",
                     args=[{"visible": [False]*len(ppm_direct_traces) + [True]*len(ppm_connecting_traces)},
                           {"title": "Price Per Minute (Connecting Flights)"}])
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

# Bar chart helper
def plotlyStackedBars(df, group_col, sub_col, title, legend_title, colors):
    if sub_col not in df.columns or df[sub_col].dropna().empty:
        st.warning(f"No valid data available for '{sub_col}'. Skipping chart.")
        return

    countDF = df.groupby([group_col, sub_col]).size().unstack(fill_value=0)
    countDF = countDF.loc[countDF.sum(axis=1).sort_values(ascending=False).index]

    fig = go.Figure()
    for i, sub_category in enumerate(countDF.columns):
        fig.add_trace(go.Bar(
            x=countDF.index,
            y=countDF[sub_category],
            name=sub_category,
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        title=title,
        barmode='stack',
        xaxis_title=group_col.capitalize(),
        yaxis_title='Number of Flights',
        legend_title=legend_title,
        xaxis_tickangle=0,
        plot_bgcolor='white',
        bargap=0.2,
        font=dict(size=12),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

# Airplane Types
st.subheader("Airplane Types by Airline")
plotlyStackedBars(
    df=directFlights,
    group_col='airline',
    sub_col='airplane',
    title='Total Flights by Airline and Airplane Type',
    legend_title='Airplane Type',
    colors=customColors
)

# Legroom
st.subheader("Legroom by Airline")
plotlyStackedBars(
    df=directFlights,
    group_col='airline',
    sub_col='legroom',
    title='Flight Legroom by Airline',
    legend_title='Legroom',
    colors=customColors
)

# WiFi (robust error handling included)
st.subheader("WiFi Availability by Airline")
plotlyStackedBars(
    df=directFlights,
    group_col='airline',
    sub_col='wifi',
    title='Flight WiFi by Airline',
    legend_title='WiFi Availability',
    colors=customColors
)

