import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ----------------------
# PAGE SETUP
# ----------------------
st.set_page_config(layout="wide")
st.title("Flights from NYC to CH")

# ----------------------
# DATA LOADING & CLEANING
# ----------------------
df = pd.read_csv("all_flights.csv")

# Convert columns
df['departureTime'] = pd.to_datetime(df['departureTime'], errors='coerce')
df['arrivalAirportTime'] = pd.to_datetime(df['arrivalAirportTime'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['durationMinutes'] = pd.to_numeric(df['durationTime'], errors='coerce')
df['totalDurationMinutes'] = pd.to_numeric(df['totalTripDuration'], errors='coerce')
df['carbonEmissionsThisFlight'] = pd.to_numeric(df.get('carbonEmissionsThisFlight'), errors='coerce')

# Clean legroom values
legroomOptions = [f"{i} inches" for i in range(28, 34)] + ["Extra reclining seat"]

def formatLegroom(val):
    try:
        return f"{int(float(val))} inches"
    except:
        return str(val)

df['legroom'] = df['legroom'].apply(formatLegroom)
df['legroom'] = pd.Categorical(df['legroom'], categories=sorted(legroomOptions), ordered=True)

# Extract features from extensions if present
if 'extentions' in df.columns:
    df['extentions'] = df['extentions'].fillna(',')
    splitExt = df['extentions'].str.split(',', n=2, expand=True).apply(lambda col: col.str.strip())
    df['recliningAndLegroom'] = splitExt[0]
    df['wifi'] = splitExt[1]
    df['carbonEmssionsEstimate'] = splitExt[2]

# Derived features
df['pricePerMinute'] = df['price'] / df['totalDurationMinutes']
df['carbonDifferencePercent'] = (
    (df['carbonEmissionsThisFlight'] - df['carbonEmissionsThisFlight'].mean()) /
    df['carbonEmissionsThisFlight'].mean() * 100
)

# Define airlines

directAirlines = ['Delta', 'SWISS', 'United']
lufthansaGroup = ['Air Dolomiti', 'Austrian', 'Brussels Airlines', 'Discover Airlines', 'Edelweiss Air', 'Eurowings', 'ITA', 'Lufthansa']
starAlliance = sorted(['Aegean', 'Air Canada', 'Air China', 'Air India', 'Air New Zealand', 'ANA', 'Asiana Airlines', 'Austrian', 'Avianca', 'Brussels Airport', 'CopaAirlines', 'Croatia Airlines', 'Egyptair', 'Ethiopian Airlines', 'Eva Air', 'LOT Polish Airlines', 'Lufthansa', 'Shenzhen Airlines', 'Singapore Airlines', 'South African Airways', 'SWISS', 'Tap Air Portugal', 'Thai', 'Turkish Airlines', 'United'])

# Toggle for connected flights
showConnected = st.toggle("Include All Airlines", value=False)
if not showConnected:
    filteredAirlines = directAirlines
else:
    filterChoice = st.selectbox("Select airlines to view:", options=['Airlines That Fly Both Direct and Connecting', 'Lufthansa Group', 'Star Alliance'])
    if filterChoice == 'Lufthansa Group':
        filteredAirlines = lufthansaGroup
    elif filterChoice == 'Star Alliance':
        filteredAirlines = starAlliance
    else:
        filteredAirlines = directAirlines

# Filter dataframe
df = df[df['airline'].isin(filteredAirlines)].copy()

# Define airport sets
nycAirports = ["JFK", "EWR", "LGA"]
swissAirports = ["ZRH", "GVA", "BSL"]

def classifyFlightType(row):
    return 'Direct' if row['departureAirportID'] in nycAirports and row['arrivalAirportID'] in swissAirports else 'Connecting'

df['flightType'] = df.apply(classifyFlightType, axis=1)

directFlights = df[df['flightType'] == 'Direct'].copy()
connectingFlights = df[df['flightType'] == 'Connecting'].copy()

# ----------------------
# COLORS
# ----------------------
customColors = ['#d71920', '#00235f', '#f9ba00', '#660000', '#000000', '#3366ff']

airlineColors = {
    'Delta': '#00235f',
    'SWISS': '#d71920',
    'United': '#1a75ff',
    'Lufthansa': '#FFD700',
    'Edelweiss Air': '#800080'
}

# ----------------------
# CHART HELPERS
# ----------------------
def plotlyStackedBars(directDF, connectingDF, group_col, sub_col, legend_title, colors):
    def buildCount(df):
        counts = df.groupby([group_col, sub_col]).size().unstack(fill_value=0)
        if hasattr(df[sub_col], 'cat'):
            for cat in df[sub_col].cat.categories:
                if cat not in counts.columns:
                    counts[cat] = 0
        return counts[sorted(counts.columns)]

    directCount = buildCount(directDF)
    connectingCount = buildCount(connectingDF)

    fig = go.Figure()
    directTraces, connectingTraces = [], []

    for i, sub_category in enumerate(directCount.columns):
        fig.add_trace(go.Bar(
            x=sorted(directCount.index),
            y=directCount[sub_category].reindex(sorted(directCount.index)),
            name=sub_category,
            marker_color=colors[i % len(colors)],
            visible=True
        ))
        directTraces.append(True)
        connectingTraces.append(False)

    for i, sub_category in enumerate(connectingCount.columns):
        fig.add_trace(go.Bar(
            x=sorted(connectingCount.index),
            y=connectingCount[sub_category].reindex(sorted(connectingCount.index)),
            name=sub_category,
            marker_color=colors[i % len(colors)],
            visible=False
        ))
        directTraces.append(False)
        connectingTraces.append(True)

    fig.update_layout(
        barmode='stack',
        xaxis_title=group_col.capitalize(),
        yaxis_title='Number of Flights',
        legend_title=legend_title,
        xaxis_tickangle=0,
        plot_bgcolor='white',
        bargap=0.2,
        font=dict(size=12),
        height=500,
        legend=dict(borderwidth=0),
        updatemenus=[
            dict(
                active=0,
                buttons=[
                    dict(label="Direct Flights", method="update", args=[{"visible": directTraces}]),
                    dict(label="Connecting Flights", method="update", args=[{"visible": connectingTraces}])
                ],
                direction="down",
                showactive=True,
                x=0.5,
                xanchor="center",
                y=1.15,
                yanchor="top"
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True)


def plotHeatmap(directDF, connectingDF, valueCol, xaxisTitle, colorscale='Blues', width=800, height=500):
    def buildHeatmapData(df):
        df_clean = df[[valueCol, 'airline']].dropna()
        binned_col = pd.cut(df_clean[valueCol], bins=10)
        pivot = df_clean.groupby(['airline', binned_col]).size().unstack(fill_value=0)
        pivot['Total'] = pivot.sum(axis=1)
        pivot = pivot.sort_values("Total", ascending=False).drop(columns="Total")
        return pivot

    directData = buildHeatmapData(directDF)
    connectingData = buildHeatmapData(connectingDF)

    trace_direct = go.Heatmap(
        z=directData.values,
        x=[str(interval) for interval in directData.columns],
        y=sorted(directData.index),
        colorscale=colorscale,
        colorbar=dict(title='Number of Flights'),
        visible=True
    )

    trace_connecting = go.Heatmap(
        z=connectingData.values,
        x=[str(interval) for interval in connectingData.columns],
        y=sorted(connectingData.index),
        colorscale=colorscale,
        colorbar=dict(title='Number of Flights'),
        visible=False
    )

    fig = go.Figure(data=[trace_direct, trace_connecting])
    fig.update_layout(
        title=xaxisTitle,
        xaxis_title=xaxisTitle,
        yaxis_title='Airline',
        template='plotly_white',
        width=width,
        height=height,
        updatemenus=[
            dict(
                active=0,
                buttons=[
                    dict(label="Direct Flights", method="update", args=[{"visible": [True, False]}, {"title": xaxisTitle + " (Direct)"}]),
                    dict(label="Connecting Flights", method="update", args=[{"visible": [False, True]}, {"title": xaxisTitle + " (Connecting)"}])
                ],
                direction="down",
                showactive=True,
                x=0.5,
                xanchor="center",
                y=1.15,
                yanchor="top"
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------
# USAGE (bubble + heatmaps)
# ----------------------
def plotBubbleChart(directDF, connectingDF, airline_col, metric_col, yaxis_title, width=800, height=500):
    def buildBubble(df):
        countDF = df.groupby([airline_col, metric_col]).size().reset_index(name='count')
        countDF = countDF.sort_values('count', ascending=False)
        return countDF

    directData = buildBubble(directDF)
    connectingData = buildBubble(connectingDF)

    traceDirect = go.Scatter(
        x=directData[airline_col],
        y=directData[metric_col],
        mode='markers',
        text=directData['count'],
        marker=dict(
            size=directData['count'],
            color=directData[metric_col],
            colorscale='RdBu',
            showscale=True,
            sizemode='area',
            sizeref=2. * directData['count'].max() / (100 ** 2),
            sizemin=4
        ),
        visible=True
    )

    traceConnecting = go.Scatter(
        x=connectingData[airline_col],
        y=connectingData[metric_col],
        mode='markers',
        text=connectingData['count'],
        marker=dict(
            size=connectingData['count'],
            color=connectingData[metric_col],
            colorscale='RdBu',
            showscale=True,
            sizemode='area',
            sizeref=2. * connectingData['count'].max() / (100 ** 2),
            sizemin=4
        ),
        visible=False
    )

    fig = go.Figure(data=[traceDirect, traceConnecting])
    fig.update_layout(
        xaxis_title='Airline',
        yaxis_title=yaxis_title,
        template='plotly_white',
        showlegend=False,
        width=width,
        height=height,
        updatemenus=[
            dict(
                active=0,
                buttons=[
                    dict(label="Direct Flights", method="update", args=[{"visible": [True, False]}]),
                    dict(label="Connecting Flights", method="update", args=[{"visible": [False, True]}])
                ],
                direction="down",
                showactive=True,
                x=0.5,
                xanchor="center",
                y=1.15,
                yanchor="top"
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True)
