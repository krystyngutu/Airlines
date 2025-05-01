import pandas as pd
import streamlit as st
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
includedAirlines = ['SWISS', 'Lufthansa', 'Edelweiss Air', 'Delta', 'United']

# Filter to only include selected airlines
df = df[df['airline'].isin(includedAirlines)].copy()

# Define new airline colors
airline_colors = {
    'Lufthansa': '#FFD700',           # gold
    'SWISS': '#d71920',               # red
    'Delta': '#00235f',               # dark blue
    'United': '#1a75ff',              # light blue
    'Edelweiss Air': '#800080'        # purple
}

# Derived column
df['pricePerMinute'] = df['price'] / df['durationMinutes']
df['carbonDifferencePercent'] = ((df['carbonEmissionsThisFlight'] - df['carbonEmissionsThisFlight'].mean()) / df['carbonEmissionsThisFlight'].mean()) * 100

# Label flights as Direct or Connecting
def classifyFlightType(row):
    if row['departureAirportID'] in nycAirports and row['arrivalAirportID'] in swissAirports:
        return 'Direct'
    return 'Connecting'

df['flightType'] = df.apply(classifyFlightType, axis=1)

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

fig = go.Figure()

# Add direct traces (visible)
for trace in direct_traces:
    trace.visible = True
    fig.add_trace(trace)

# Add connecting traces (hidden)
for trace in connecting_traces:
    trace.visible = False
    fig.add_trace(trace)

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
carbon_fig = go.Figure()

# Add direct traces (visible)
for trace in carbon_direct_traces:
    trace.visible = True
    carbon_fig.add_trace(trace)

# Add connecting traces (hidden initially)
for trace in carbon_connecting_traces:
    trace.visible = False
    carbon_fig.add_trace(trace)

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
st.subheader('Direct Flight Visuals')
plotlyStackedBars(
    df=directFlights,
    group_col='airline',
    sub_col='airplane',
    title='Airplane Types by Airline',
    legend_title='Airplane Type',
    colors=customColors
)

# Legroom
plotlyStackedBars(
    df=directFlights,
    group_col='airline',
    sub_col='legroom',
    title='Legroom by Airline',
    legend_title='Legroom',
    colors=customColors
)

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
plotBubbleChart(
    df=directFlights,
    airline_col='airline',
    metric_col='durationTime',
    yaxis_title='Duration (min)',
    title='Flight Duration by Airline (Bubble Size = Count)',
    width=1000
)

plotBubbleChart(
    df=directFlights,
    airline_col='airline',
    metric_col='price',
    yaxis_title='Price (USD)',
    title='Flight Prices by Airline (Bubble Size = Count)'
)

plotBubbleChart(
    df=directFlights,
    airline_col='airline',
    metric_col='carbonEmissionsThisFlight',
    yaxis_title='Carbon Emissions by Airline per Flight (Bubble Size = Count)'
)


plotBubbleChart(
    df=directFlights,
    airline_col='airline',
    metric_col='carbonDifferencePercent',
    yaxis_title='Carbon Difference (%) by Airline per Flight (Bubble Size = Count)'
)

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
st.subheader("DIRECT FLIGHT: Carbon Difference Percent by Airline")
plotHeatmap(
    directFlights,
    valueCol='carbonDifferencePercent',
    title='Carbon Difference Percent by Airline',
    xaxisTitle='Carbon Difference Percent',
    colorscale='Reds'
)

st.subheader("DIRECT FLIGHTS: Price by Airline")
plotHeatmap(
    directFlights,
    valueCol='price',
    title='Price by Airline',
    xaxisTitle='Price (USD)',
    colorscale='Reds'
)

st.subheader("DIRECT FLIGHTS: Duration Time by Airline")
plotHeatmap(
    directFlights,
    valueCol='durationTime',
    title='Duration Time by Airline',
    xaxisTitle='Duration (min)',
    colorscale='Reds'
)
