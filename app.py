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

# Subset
directFlights = df[
    (df["departureAirportID"].isin(nycAirports)) &
    (df["arrivalAirportID"].isin(swissAirports))
].copy()

directFlights['legroom'] = directFlights['legroom'].fillna("Extra reclining seat")
if "recliningAndLegroom" in directFlights.columns:
    directFlights.drop(columns=["recliningAndLegroom"], inplace=True)

# Classify flight type
def classify_flight_type(row):
    if row['airline'] in ['Delta', 'United', 'SWISS']:
        return 'Direct'
    return 'Connecting'

directFlights['flightType'] = directFlights.apply(classify_flight_type, axis=1)

st.title("Flights from NYC to CH")

# Filtered data
filtered = directFlights.copy()

# PRICE OVER TIME - BY AIRLINE & FLIGHT TYPE
st.subheader("Price Over Time by Airline and Flight Type")
filtered = filtered.sort_values('departureTime')

# Define airline colors
airline_colors = {
    'Delta': 'navy',
    'SWISS': 'red',
    'United': 'lightblue'
}

# Split by flight type
direct_df = filtered[filtered['flightType'] == 'Direct']
connecting_df = filtered[filtered['flightType'] == 'Connecting']

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

# Create traces
direct_traces = create_traces(direct_df)
connecting_traces = create_traces(connecting_df)

# Build figure
fig = go.Figure(data=direct_traces)
for trace in connecting_traces:
    fig.add_trace(trace)

# Dropdown toggle for flight type
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
    ]
)

fig.update_layout(
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

st.plotly_chart(fig, use_container_width=True)

# Carbon Emissions vs Price
st.subheader("Carbon Emissions vs Price")
fig2 = px.scatter(filtered, x='carbonEmissionsThisFlight', y='price', hover_data=['departureTime'])
st.plotly_chart(fig2, use_container_width=True)

# Price Per Minute
st.subheader("Price Per Minute")
fig3 = px.line(filtered.sort_values('departureTime'), x='departureTime', y='pricePerMinute')
st.plotly_chart(fig3, use_container_width=True)

# Histograms
col1, col2 = st.columns(2)
with col1:
    st.subheader("Histogram: Prices")
    fig4 = px.histogram(filtered, x='price')
    st.plotly_chart(fig4, use_container_width=True)

with col2:
    st.subheader("Histogram: Duration")
    fig5 = px.histogram(filtered, x='durationMinutes')
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
