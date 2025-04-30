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
included_airlines = ['SWISS', 'Delta', 'United', 'Lufthansa']
df = df[df['airline'].isin(included_airlines)].copy()

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
def plotHeatmap(df, x_col, y_col, z_col, title):
    heatmap_df = df.groupby([x_col, y_col])[z_col].mean().reset_index()
    heatmap_pivot = heatmap_df.pivot(index=y_col, columns=x_col, values=z_col)
    fig = px.imshow(
        heatmap_pivot,
        color_continuous_scale='RdBu_r',
        labels=dict(color=z_col),
        title=title
    )
    st.plotly_chart(fig, use_container_width=True)

# Insert heatmaps here
st.subheader("Heatmap: Price by Airline and Legroom")
plotHeatmap(directFlights, x_col='airline', y_col='legroom', z_col='price', title='Average Price by Airline and Legroom')

st.subheader("Heatmap: Carbon Emissions by Airline and Airplane")
plotHeatmap(directFlights, x_col='airline', y_col='airplane', z_col='carbonEmissionsThisFlight', title='Carbon Emissions by Airline and Airplane')

# Bubble chart helper function
def plotBubbleChart(df, airline_col, metric_col, yaxis_title, chart_title, 
                    width=800, height=500):
    countDF = df.groupby([airline_col, metric_col]).size().reset_index(name='count')
    countDF = countDF.sort_values('count', ascending=False)
    priorityOrder = ['SWISS', 'United', 'Delta']
    allAirlines = countDF[airline_col].unique()
    remainingAirlines = sorted([a for a in allAirlines if a not in priorityOrder])
    fullOrder = priorityOrder + remainingAirlines
    countDF[airline_col] = pd.Categorical(countDF[airline_col], categories=fullOrder, ordered=True)
    countDF = countDF.sort_values(airline_col)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=countDF[airline_col],
        y=countDF[metric_col],
        mode='markers+text' if metric_col == 'durationTime' else 'markers',
        text=countDF['count'],
        marker=dict(
            size=countDF['count'],
            color=countDF[metric_col],
            colorscale='RdBu',
            showscale=True,
            sizemode='area',
            sizeref=2. * countDF['count'].max() / (100 ** 2),
            sizemin=4
        )
    ))

    fig.update_layout(
        title=dict(text=chart_title, x=0.5, xanchor='center'),
        xaxis_title='Airline',
        yaxis_title=yaxis_title,
        xaxis_tickangle=0,
        template='plotly_white',
        showlegend=False,
        width=width,
        height=height
    )

    st.plotly_chart(fig, use_container_width=True)

# Bubble charts section
st.subheader("Flight Duration vs Airline (Bubble Size = Count)")
plotBubbleChart(
    df=directFlights,
    airline_col='airline',
    metric_col='durationTime',
    yaxis_title='Duration (min)',
    chart_title='Flight Duration vs Airline (Bubble Size = Count)',
    width=1000
)

st.subheader("Flight Prices vs Airline (Bubble Size = Count)")
plotBubbleChart(
    df=directFlights,
    airline_col='airline',
    metric_col='price',
    yaxis_title='Price (USD)',
    chart_title='Flight Prices vs Airline (Bubble Size = Count)'
)

st.subheader("Flight Carbon Emissions vs Airline (Bubble Size = Count)")
plotBubbleChart(
    df=directFlights,
    airline_col='airline',
    metric_col='carbonEmissionsThisFlight',
    yaxis_title='Carbon Emissions This Flight',
    chart_title='Flight Carbon Emissions vs Airline (Bubble Size = Count)'
)

st.subheader("Carbon Percent Difference vs Airline (Bubble Size = Count)")
plotBubbleChart(
    df=directFlights,
    airline_col='airline',
    metric_col='carbonDifferencePercent',
    yaxis_title='Carbon Difference Percent This Flight',
    chart_title='Carbon Percent Difference (to the Average) vs Airline (Bubble Size = Count)'
)
