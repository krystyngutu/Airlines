import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Load data
st.set_page_config(layout='wide')
df = pd.read_csv('all_flights.csv')

# Clean and preprocess
df['departureTime'] = pd.to_datetime(df['departureTime'], errors='coerce')
df['arrivalAirportTime'] = pd.to_datetime(df['arrivalAirportTime'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['durationMinutes'] = pd.to_numeric(df['durationTime'], errors='coerce')
df['carbonEmissionsThisFlight'] = pd.to_numeric(df.get('carbonEmissionsThisFlight'), errors='coerce')

# Define  airports and airlines to look at
nycAirports = ['JFK', 'EWR', 'LGA']
swissAirports = ['ZRH', 'GVA', 'BSL']

includedAirlines = ['SWISS', 'Lufthansa', 'Delta', 'United']

# Filter to only include selected airlines
df = df[df['airline'].isin(includedAirlines)].copy()

# Label flights as Direct or Connecting
def classifyFlightType(row):
    if row['departureAirportID'] in nycAirports and row['arrivalAirportID'] in swissAirports:
        return 'Direct'
    return 'Connecting'

df['flightType'] = df.apply(classifyFlightType, axis=1)

# Apply following changes only to direct flights
directMask = df['flightType'] == 'Direct'

df.loc[directMask, 'legroom'] = df.loc[directMask, 'legroom'].fillna("Extra reclining seat")

if "recliningAndLegroom" in df.columns:
    df.drop(columns=["recliningAndLegroom"], inplace=True)

# Split into direct and connecting flights
directFlights = df[df['flightType'] == 'Direct'].copy()
connectingFlights = df[df['flightType'] == 'Connecting'].copy()

# Custom color palette
customColors = [
    '#d71920',                                    # red
    '#00235f',                                    # dark blue
    '#f9ba00',                                    # gold
    '#660000',                                    # burgundy
    '#000000',                                    # black
    '#3366ff',                                    # blue
    '#ffffff'                                     # white
]

# Define new airline colors
airlineColors = {
    'Lufthansa': '#ffd700',                       # gold
    'SWISS': '#d71920',                           # red
    'Delta': '#00235f',                           # dark blue
    'United': '#1a75ff',                          # light blue
}

st.title("Flights from NYC to CH")

# Create traces for graphs
def createTraces(df):
    traces = []
    airlines = ['Lufthansa', 'United', 'Delta']      # SWISS excluded for now

    for airline in airlines:
        if airline in df['airline'].unique():
            data = df[df['airline'] == airline]
            traces.append(go.Scatter(
                x=data['departureTime'],
                y=data['price'],
                mode='markers+lines',
                name=airline,
                hovertext=data['flightNumber'],
                marker=dict(color=airlineColors.get(airline, 'gray')),
                legendgroup=airline,
                legendrank=['SWISS', 'Lufthansa', 'United', 'Delta'].index(airline)
            ))

    # Add SWISS last to bring it to the top of the plot visually
    if 'SWISS' in df['airline'].unique():
        data = df[df['airline'] == 'SWISS']
        traces.append(go.Scatter(
            x=data['departureTime'],
            y=data['price'],
            mode='markers+lines',
            name='SWISS',
            hovertext=data['flightNumber'],
            marker=dict(color=airlineColors.get('SWISS', 'gray')),
            line=dict(width=3),                   # make SWISS line slightly thicker
            legendgroup='SWISS',
            legendrank=0                          # appears first in the legend
        ))

    return traces

# Create traces for both flight types
directTraces = createTraces(directFlights)
connectingTraces = createTraces(connectingFlights)

fig = go.Figure()

# Add direct traces (visible)
for trace in directTraces:
    trace.visible = True
    fig.add_trace(trace)

# Add connecting traces (hidden)
for trace in connectingTraces:
    trace.visible = False
    fig.add_trace(trace)

fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=[
                dict(label="Direct Flights",
                     method="update",
                     args=[{"visible": [True]*len(directTraces) + [False]*len(connectingTraces)},
                           {"title": "Price Over Time (Direct Flights)"}]),
                dict(label="Connecting Flights",
                     method="update",
                     args=[{"visible": [False]*len(directTraces) + [True]*len(connectingTraces)},
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
carbonDirectTraces = []
carbonConnectingTraces = []

for airline in df['airline'].unique():
    # Direct
    dataDirect = directFlights[directFlights['airline'] == airline]
    carbonDirectTraces.append(go.Scatter(
        x=dataDirect['carbonEmissionsThisFlight'],
        y=dataDirect['price'],
        mode='markers',
        name=airline,
        hovertext=dataDirect['flightNumber'],
        marker=dict(color=airlineColors.get(airline, 'gray'))
    ))

    # Connecting
    dataConnecting = connectingFlights[connectingFlights['airline'] == airline]
    carbonConnectingTraces.append(go.Scatter(
        x=dataConnecting['carbonEmissionsThisFlight'],
        y=dataConnecting['price'],
        mode='markers',
        name=airline,
        hovertext=dataConnecting['flightNumber'],
        marker=dict(color=airlineColors.get(airline, 'gray'))
    ))

# Combine into figure
carbonFig = go.Figure()

# Add direct traces (visible)
for trace in carbonDirectTraces:
    trace.visible = True
    carbonFig.add_trace(trace)

# Add connecting traces (hidden initially)
for trace in carbonConnectingTraces:
    trace.visible = False
    carbonFig.add_trace(trace)

carbonFig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=[
                dict(label="Direct Flights",
                     method="update",
                     args=[{"visible": [True]*len(carbonDirectTraces) + [False]*len(carbonConnectingTraces)},
                           {"title": "Carbon Emissions vs Price (Direct Flights)"}]),
                dict(label="Connecting Flights",
                     method="update",
                     args=[{"visible": [False]*len(carbonDirectTraces) + [True]*len(carbonConnectingTraces)},
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

st.plotly_chart(carbonFig, use_container_width=True)

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
st.subheader("DIRECT FLIGHTS: Airplane Types by Airline")
plotlyStackedBars(
    df=directFlights,
    group_col='airline',
    sub_col='airplane',
    legend_title='Airplane Type',
    colors=customColors
)

# Legroom
st.subheader("DIRECT FLIGHTS: Legroom by Airline")
plotlyStackedBars(
    df=directFlights,
    group_col='airline',
    sub_col='legroom',
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
st.subheader("DIRECT FLIGHTS: Flight Duration vs Airline (Bubble Size = Count)")
plotBubbleChart(
    df=directFlights,
    airline_col='airline',
    metric_col='durationMinutes',
    yaxis_title='Duration (min)',
    width=1000
)

st.subheader("DIRECT FLIGHTS: Flight Prices vs Airline (Bubble Size = Count)")
plotBubbleChart(
    df=directFlights,
    airline_col='airline',
    metric_col='price',
    yaxis_title='Price (USD)'
)

st.subheader("DIRECT FLIGHTS: Flight Carbon Emissions vs Airline (Bubble Size = Count)")
plotBubbleChart(
    df=directFlights,
    airline_col='airline',
    metric_col='carbonEmissionsThisFlight',
    yaxis_title='Carbon Emissions This Flight'
)

st.subheader("DIRECT FLIGHTS: Carbon Percent Difference vs Airline (Bubble Size = Count)")
plotBubbleChart(
    df=directFlights,
    airline_col='airline',
    metric_col='carbonDifferencePercent',
    yaxis_title='Carbon Difference Percent This Flight'
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
