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
df['legroom'] = pd.Categorical(df['legroom'], categories=legroomOptions, ordered=True)

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

# Define  airlines to include
directAirlines = ['SWISS', 'United', 'Delta']
lufthansaGroup = ['Austrian', 'Brussels Airlines', 'Discover Airlines', 'Eurowings', 'Edelweiss Air', 'ITA', 'Air Dolomiti', 'Lufthansa']
starAlliance = ['Aegean', 'Air Canada', 'Air China', 'Air India', 'Air New Zealand', 'ANA', 'Asiana Airlines', 'Austrian', 'Avianca', 'Brussels Airport', 'CopaAirlines', 'Croatia Airlines', 'Egyptair', 'Ethiopian Airlines', 'Eva Air', 'LOT Polish Airlines', 'Lufthansa', 'Shenzhen Airlines', 'Singapore Airlines', 'South African Airways', 'SWISS', 'Tap Air Portugal', 'Thai', 'Turkish Airlines', 'United']

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

# Filter DataFrame
df = df[df['airline'].isin(filteredAirlines)].copy()

# Define airports to include
nycAirports = ["JFK", "EWR", "LGA"]
swissAirports = ["ZRH", "GVA", "BSL"]

# Label flights as Direct or Connecting
def classifyFlightType(row):
    if row['departureAirportID'] in nycAirports and row['arrivalAirportID'] in swissAirports:
        return 'Direct'
    return 'Connecting'

df['flightType'] = df.apply(classifyFlightType, axis=1)

# Split into direct and connecting flights
directFlights = df[df['flightType'] == 'Direct'].copy()
connectingFlights = df[df['flightType'] == 'Connecting'].copy()

# ----------------------
# COLORS
# ----------------------
customColors = ['#d71920', '#00235f', '#f9ba00', '#660000', '#000000', '#3366ff']

# Define new airline colors
airlineColors = {
    'Lufthansa': '#FFD700',           # gold
    'SWISS': '#d71920',               # red
    'Delta': '#00235f',               # dark blue
    'United': '#1a75ff',              # light blue
    'Edelweiss Air': '#800080'        # purple
}


# ----------------------
# CHART HELPERS
# ----------------------
def createLineChart(directDF, connectingDF):
    def createTraces(df):
        traces = []
        for airline in sorted(df['airline'].dropna().unique()):
            data = df[df['airline'] == airline]
            traces.append(go.Scatter(
                x=data['departureTime'],
                y=data['price'],
                mode='markers+lines',
                name=airline,
                hovertext=data['flightNumber'],
                marker=dict(color=airlineColors.get(airline, 'gray'))
            ))
        return traces

    directTraces = createTraces(directDF)
    connectingTraces = createTraces(connectingDF)

    fig = go.Figure(data=directTraces + connectingTraces)
    for i, trace in enumerate(fig.data):
        trace.visible = i < len(directTraces)

    fig.update_layout(
        xaxis_title="Departure Date",
        yaxis_title="Price (USD)",
        legend_title_text="Airlines",
        hovermode="closest",
        height=600,
        legend=dict(
            font=dict(size=11),
            orientation="v",
            x=1,
            y=1,
            xanchor='left',
            yanchor='top',
            itemclick='toggle',
            itemdoubleclick='toggleothers'
        ),
        updatemenus=[
            dict(
                active=0,
                buttons=[
                    dict(label="Direct Flights",
                         method="update",
                         args=[{"visible": [True]*len(directTraces) + [False]*len(connectingTraces)}]),
                    dict(label="Connecting Flights",
                         method="update",
                         args=[{"visible": [False]*len(directTraces) + [True]*len(connectingTraces)}])
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
    st.subheader("Price Over Time")
    st.plotly_chart(fig, use_container_width=True)

def plotlyStackedBars(directDF, connectingDF, group_col, sub_col, legend_title, colors):
    def buildCount(df):
        counts = df.groupby([group_col, sub_col]).size().unstack(fill_value=0)
        if pd.api.types.is_categorical_dtype(df[sub_col]):
            for cat in df[sub_col].cat.categories:
                if cat not in counts.columns:
                    counts[cat] = 0
        return counts.sort_index(axis=1)

    directCount = buildCount(directDF)
    connectingCount = buildCount(connectingDF)

    fig = go.Figure()

    for i, sub_category in enumerate(directCount.columns):
        fig.add_trace(go.Bar(
            x=directCount.index,
            y=directCount[sub_category],
            name=sub_category,
            marker_color=colors[i % len(colors)],
            visible=True
        ))

    for i, sub_category in enumerate(connectingCount.columns):
        fig.add_trace(go.Bar(
            x=connectingCount.index,
            y=connectingCount[sub_category],
            name=sub_category,
            marker_color=colors[i % len(colors)],
            visible=False
        ))

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
        updatemenus=[
            dict(
                active=0,
                buttons=[
                    dict(label="Direct Flights",
                         method="update",
                         args=[{"visible": [True]*len(directCount.columns) + [False]*len(connectingCount.columns)}]),
                    dict(label="Connecting Flights",
                         method="update",
                         args=[{"visible": [False]*len(directCount.columns) + [True]*len(connectingCount.columns)}])
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
        y=directData.index,
        colorscale=colorscale,
        colorbar=dict(title='Number of Flights'),
        visible=True
    )

    traceConnecting = go.Heatmap(
        z=connectingData.values,
        x=[str(interval) for interval in connectingData.columns],
        y=connectingData.index,
        colorscale=colorscale,
        colorbar=dict(title='Number of Flights'),
        visible=False
    )

    fig = go.Figure(data=[trace_direct, traceConnecting])

    fig.update_layout(
        title=f'{xaxisTitle} by Airline',
        xaxis_title=xaxisTitle,
        yaxis_title='Airline',
        template='plotly_white',
        width=width,
        height=height,
        updatemenus=[
            dict(
                active=0,
                buttons=[
                    dict(label="Direct Flights",
                         method="update",
                         args=[{"visible": [True, False]},
                               {"title": f"{xaxisTitle} by Airline (Direct)"}]),
                    dict(label="Connecting Flights",
                         method="update",
                         args=[{"visible": [False, True]},
                               {"title": f"{xaxisTitle} by Airline (Connecting)"}])
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
