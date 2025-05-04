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

# Extract features from extensions if present
if 'extensions' in df.columns:
    df['extensions'] = df['extensions'].fillna(',')
    splitExt = df['extensions'].str.split(',', n=2, expand=True).apply(lambda col: col.str.strip())
    df['recliningAndLegroom'] = splitExt[0]
    df['wifi'] = splitExt[1]
    df['carbonEmssionsEstimate'] = splitExt[2]

def extractParensOrKeep(val):
    if pd.isna(val):
        return val
    import re
    match = re.search(r'\((.*?)\)', val)
    return match.group(1) if match else val.strip()

df['recliningAndLegroom'] = df['recliningAndLegroom'].apply(extractParensOrKeep)
df['legroom'] = df['legroom'].fillna(df['recliningAndLegroom'])
df.loc[df['wifi'].str.startswith('Carbon', na=False), 'wifi'] = 'Status Unknown'

# Derived features
df['pricePerMinute'] = df['price'] / df['totalDurationMinutes']
df['carbonDifferencePercent'] = (
    (df['carbonEmissionsThisFlight'] - df['carbonEmissionsThisFlight'].mean()) /
    df['carbonEmissionsThisFlight'].mean() * 100
)

# Define  airlines to include
directAirlines = ['SWISS', 'United', 'Delta']
lufthansaGroup = ['Austrian', 'Brussels Airlines', 'Discover Airlines', 'Eurowings', 'Edelweiss Air', 'ITA', 'Air Dolomiti', 'Lufthansa', 'SWISS']
starAlliance = ['Aegean', 'Air Canada', 'Air China', 'Air India', 'Air New Zealand', 'ANA', 'Asiana Airlines', 'Austrian', 'Avianca', 'Brussels Airport', 'CopaAirlines', 'Croatia Airlines', 'Egyptair', 'Ethiopian Airlines', 'Eva Air', 'LOT Polish Airlines', 'Lufthansa', 'Shenzhen Airlines', 'Singapore Airlines', 'South African Airways', 'SWISS', 'Tap Air Portugal', 'Thai', 'Turkish Airlines', 'United']

# Dropdown: default = Direct for general, Connecting for LHG and Star Alliance
filterOptions = ['Airlines That Fly Both Direct and Connecting', 'Lufthansa Group', 'Star Alliance']
defaultIndex = 0  # Default to general group

filterChoice = st.selectbox("Select airlines to view:", options=filterOptions, index=defaultIndex)

# Set filtered airline list and default flight type
if filterChoice == 'Lufthansa Group':
    filteredAirlines = lufthansaGroup
    showDirect = False
    showConnecting = True
elif filterChoice == 'Star Alliance':
    filteredAirlines = starAlliance
    showDirect = False
    showConnecting = True
else:
    filteredAirlines = directAirlines
    showDirect = True
    showConnecting = False

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
customColors = ['#d71920', '#00235f', '#f9ba00', '#660000', '#800080', '#3366ff',
                '#c3f550', '#fbaa3f', '#000000']

# Define new airline colors
airlineColors = {
    'Lufthansa': '#ffd700',           # gold
    'SWISS': '#d71920',               # red
    'Delta': '#00235f',               # dark blue
    'United': '#1a75ff',              # light blue
    'Edelweiss Air': '#800080',       # purple
    'Air Dolomiti': '#32cd32',        # lime green
    'Austrian': '#c3f550',            # lime
    'ITA': '#fbaa3f',                 # orange
    'Brussels Airlines': '#00235f',   # dark blue
    'Eurowings': '#1a75ff',           # light blue
    'Aegean': '#767676',              # gray
    'Air Canada': '#00235f',          # dark blue
    'Tap Air Portugal': '#fbaa3f',    # orange
    'Turkish Airlines': '#800080'     # purple    
}

# ----------------------
# PRICE COMPARISON PLOTS
# ----------------------

# Drop rows with missing values for key comparisons
priceDF = df.dropna(subset=['price', 'durationMinutes', 'carbonEmissionsThisFlight', 'legroom', 'travelClass', 'airplane'])

# Standardize aircraft types for connecting flights
def classifyAircraft(aircraft):
    if pd.isna(aircraft):
        return "Other"
    aircraft = aircraft.lower()
    if aircraft.startswith("airbus"):
        return "Airbus"
    elif aircraft.startswith("boeing"):
        return "Boeing"
    elif aircraft.startswith("canadair"):
        return "Canadair"
    elif aircraft.startswith("embraer"):
        return "Embraer"
    else:
        return "Other"

# 1. Price vs Duration (scatter)
st.subheader("Price vs Duration")
st.plotly_chart(go.Figure(
    data=[go.Scatter(
        x=priceDF['totalDurationMinutes'],
        y=priceDF['price'],
        mode='markers',
        marker=dict(color='blue'),
        name='Duration'
    )],
    layout=go.Layout(
        xaxis_title="Duration (minutes)",
        yaxis_title="Price (USD)",
        height=450
    )
), use_container_width=True)

# 2. Price vs Carbon Emissions (scatter)
st.subheader("Price vs Carbon Emissions")
st.plotly_chart(go.Figure(
    data=[go.Scatter(
        x=priceDF['carbonEmissionsThisFlight'],
        y=priceDF['price'],
        mode='markers',
        marker=dict(color='green'),
        name='Carbon Emissions'
    )],
    layout=go.Layout(
        xaxis_title="Carbon Emissions (kg CO₂)",
        yaxis_title="Price (USD)",
        height=450
    )
), use_container_width=True)

# 3. Price vs Legroom (box)
st.subheader("Price by Legroom")
st.plotly_chart(go.Figure(
    data=[go.Box(
        x=priceDF['legroom'],
        y=priceDF['price'],
        name='Legroom',
        marker_color='orange'
    )],
    layout=go.Layout(
        xaxis_title="Legroom (inches or category)",
        yaxis_title="Price (USD)",
        height=450
    )
), use_container_width=True)

# 4. Price vs Travel Class (box)
st.subheader("Price by Travel Class")
st.plotly_chart(go.Figure(
    data=[go.Box(
        x=priceDF['travelClass'],
        y=priceDF['price'],
        name='Travel Class',
        marker_color='purple'
    )],
    layout=go.Layout(
        xaxis_title="Travel Class",
        yaxis_title="Price (USD)",
        height=450
    )
), use_container_width=True)

# 5. Price vs Aircraft (box)
st.subheader("Price by Aircraft Type")
st.plotly_chart(go.Figure(
    data=[go.Box(
        x=priceDF['airplane'],
        y=priceDF['price'],
        name='Aircraft',
        marker_color='darkred'
    )],
    layout=go.Layout(
        xaxis_title="Aircraft Type",
        yaxis_title="Price (USD)",
        height=450
    )
), use_container_width=True)

# ----------------------
# CHART HELPERS
# ----------------------
def createTraces(df):
    traces = []
    for airline in sorted(df['airline'].unique()):
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

# Create traces for direct and connecting flights
directTraces = createTraces(directFlights)
connectingTraces = createTraces(connectingFlights)

fig = go.Figure()

# Add direct traces (visible)
for trace in directTraces:
    trace.visible=showDirect
    fig.add_trace(trace)

# Add connecting traces (hidden)
for trace in connectingTraces:
    trace.visible=showConnecting
    fig.add_trace(trace)

fig.update_layout(
    updatemenus=[
        dict(
            active=0 if showDirect else 1,
            buttons=[
                dict(label='Direct Flights',
                     method='update',
                     args=[{'visible':[True]*len(directTraces) + [False]*len(connectingTraces)}]),
                dict(label='Connecting Flights',
                     method='update',
                     args=[{'visible':[False]*len(directTraces) + [True]*len(connectingTraces)}])
            ],
            direction='down',
            showactive=True,
            x=0.5,
            xanchor='center',
            y=1.1,
            yanchor='top'
        )
    ],
    xaxis_title="Departure Date",
    yaxis_title="Price (USD)",
    legend_title_text="Airline",
    hovermode="closest",
    height=600,
    legend=dict(
        title_font=dict(size=12),
        font=dict(size=11),
        orientation="v",
        x=1,
        y=1,
        xanchor='left',
        yanchor='top',
        borderwidth=1,
        itemclick='toggle',
        itemdoubleclick='toggleothers'
    )
)

st.subheader("Price Over Time")
st.plotly_chart(fig, use_container_width=True)

# ---------- Carbon Emissions vs Price ----------
st.subheader("Price vs Carbon Emissions")

# Prepare traces
carbonDirectTraces = []
carbonConnectingTraces = []

for airline in sorted(df['airline'].unique()):
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
    trace.visible=showDirect
    carbonFig.add_trace(trace)

# Add connecting traces (hidden initially)
for trace in carbonConnectingTraces:
    trace.visible=showConnecting
    carbonFig.add_trace(trace)

carbonFig.update_layout(
    updatemenus=[
        dict(
            active=0 if showDirect else 1,
            buttons=[
                dict(label="Direct Flights",
                     method="update",
                     args=[{"visible": [True]*len(carbonDirectTraces) + [False]*len(carbonConnectingTraces)}]),
                dict(label="Connecting Flights",
                     method="update",
                     args=[{"visible": [False]*len(carbonDirectTraces) + [True]*len(carbonConnectingTraces)}])
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
        borderwidth=1
    )
)

st.plotly_chart(carbonFig, use_container_width=True)

# Bar chart helper with toggle for Direct vs Connecting
def plotlyStackedBars(directDF, connectingDF, group_col, sub_col, legend_title, colors):
    # Determine default view
    filterChoice = st.session_state.get('filterChoice', 'Airlines That Fly Both Direct and Connecting')
    showDirect = filterChoice == 'Airlines That Fly Both Direct and Connecting'
    showConnecting = not showDirect

    def buildCount(df):
        if not pd.api.types.is_categorical_dtype(df[sub_col]):
            df[sub_col] = pd.Categorical(df[sub_col])  # Ensure consistency
        counts = df.groupby([group_col, sub_col]).size().unstack(fill_value=0)

        for cat in df[sub_col].cat.categories:
            if cat not in counts.columns:
                counts[cat] = 0

        counts = counts.reindex(sorted(counts.columns), axis=1)
        return counts

    directCount = buildCount(directDF)
    connectingCount = buildCount(connectingDF)

    fig = go.Figure()
    directTraces = []
    connectingTraces = []

    # Add direct traces
    for i, sub_category in enumerate(directCount.columns):
        trace = go.Bar(
            x=directCount.index,
            y=directCount[sub_category],
            name=f'{sub_category}',
            marker_color=colors[i % len(colors)],
            visible=showDirect,
            legendgroup=f'{sub_category}',
            showlegend=True
        )
        fig.add_trace(trace)
        directTraces.append(True)
        connectingTraces.append(False)

    # Add connecting traces
    for i, sub_category in enumerate(connectingCount.columns):
        trace = go.Bar(
            x=connectingCount.index,
            y=connectingCount[sub_category],
            name=f'{sub_category}',
            marker_color=colors[i % len(colors)],
            visible=showConnecting,
            legendgroup=f'{sub_category}',
            showlegend=True
        )
        fig.add_trace(trace)
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
        updatemenus=[
            dict(
                active=0 if showDirect else 1,
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

directFlights['airplane'] = directFlights['airplane'].apply(classifyAircraft)
connectingFlights['airplane'] = connectingFlights['airplane'].apply(classifyAircraft)

# Aircraft breakdown
st.subheader('Aircraft by Airline')
plotlyStackedBars(
    directFlights,
    connectingFlights,
    group_col='airline',
    sub_col='airplane',
    legend_title='Aircraft',
    colors=customColors
)

# Legroom and Reclining breakdown
st.subheader('Legroom by Airline')
plotlyStackedBars(
    directFlights,
    connectingFlights,
    group_col='airline',
    sub_col='legroom',
    legend_title='Legroom',
    colors=customColors
)

# Get all unique wifi categories from both DataFrames
wifiCategories = sorted(set(directFlights['wifi'].dropna().unique()).union(connectingFlights['wifi'].dropna().unique()))

# Convert to ordered categorical
directFlights['wifi'] = pd.Categorical(directFlights['wifi'], categories=wifiCategories, ordered=True)
connectingFlights['wifi'] = pd.Categorical(connectingFlights['wifi'], categories=wifiCategories, ordered=True)

# WiFi breakdown
st.subheader('WiFi by Airline')
plotlyStackedBars(
    directFlights,
    connectingFlights,
    group_col='airline',
    sub_col='wifi',
    legend_title='WiFi',
    colors=customColors
)

# Travel Class breakdown
st.subheader('Travel Class by Airline')
plotlyStackedBars(
    directFlights,
    connectingFlights,
    group_col='airline',
    sub_col='travelClass',
    legend_title='Travel Class',
    colors=customColors
)

# Bubble chart helper function with flight type toggle
def plotBubbleChart(directDF, connectingDF, airline_col, metric_col, yaxis_title, width=800, height=500):
    # Determine which flight type to show by default
    filterChoice = st.session_state.get('filterChoice', 'Airlines That Fly Both Direct and Connecting')
    showDirect = filterChoice == 'Airlines That Fly Both Direct and Connecting'
    showConnecting = not showDirect

    def buildBubble(df):
        df['airline'] = pd.Categorical(df['airline'], categories=sorted(df['airline'].unique()), ordered=True)
        countDF = df.groupby(['airline', metric_col]).size().reset_index(name='count')
        countDF = countDF.sort_values('airline')  # Alphabetical order
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
        visible=showDirect
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
        visible=showConnecting
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
                active=0 if showDirect else 1,
                buttons=[
                    dict(label="Direct Flights",
                         method="update",
                         args=[{"visible": [True, False]}]),
                    dict(label="Connecting Flights",
                         method="update",
                         args=[{"visible": [False, True]}])
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

# Bubble charts
# Flight duration breakdown
st.subheader('Total Duration')
plotBubbleChart(directFlights, connectingFlights, 'airline', 'durationTime', 'Duration (min)', width=1000)

# Flight prices breakdown
st.subheader('Prices')
plotBubbleChart(directFlights, connectingFlights, 'airline', 'price', 'Price (USD)')

# Carbon emissions breakdown
st.subheader('Carbon Emissions')
plotBubbleChart(directFlights, connectingFlights, 'airline', 'carbonEmissionsThisFlight', 'Carbon Emissions')

# Carbon difference breakdown
st.subheader('Carbon Difference')
plotBubbleChart(directFlights, connectingFlights, 'airline', 'carbonDifferencePercent', 'Carbon Difference')
# Heatmap helper function with flight type toggle (no internal title)
def plotHeatmap(directDF, connectingDF, valueCol, colorscale='Blues', width=800, height=500):
    # Determine toggle state
    filterChoice = st.session_state.get('filterChoice', 'Airlines That Fly Both Direct and Connecting')
    showDirect = filterChoice == 'Airlines That Fly Both Direct and Connecting'
    showConnecting = not showDirect

    def buildHeatmapData(df):
        df_clean = df[[valueCol, 'airline']].dropna()
        if df_clean.empty:
            return pd.DataFrame()

        # Sort airline names alphabetically
        df_clean['airline'] = df_clean['airline'].astype(str)
        airline_order = sorted(df_clean['airline'].unique())
        df_clean['airline'] = pd.Categorical(df_clean['airline'], categories=airline_order, ordered=True)

        binned_col = pd.cut(df_clean[valueCol], bins=10)
        pivot = df_clean.groupby(['airline', binned_col]).size().unstack(fill_value=0)
        pivot = pivot.sort_index(level=0)
        return pivot

    directData = buildHeatmapData(directDF)
    connectingData = buildHeatmapData(connectingDF)

    traceDirect = go.Heatmap(
        z=directData.values if not directData.empty else [[0]],
        x=[str(col) for col in directData.columns] if not directData.empty else [],
        y=directData.index.tolist() if not directData.empty else [],
        colorscale=colorscale,
        colorbar=dict(title='Number of Flights'),
        visible=showDirect
    )

    traceConnecting = go.Heatmap(
        z=connectingData.values if not connectingData.empty else [[0]],
        x=[str(col) for col in connectingData.columns] if not connectingData.empty else [],
        y=connectingData.index.tolist() if not connectingData.empty else [],
        colorscale=colorscale,
        colorbar=dict(title='Number of Flights'),
        visible=showConnecting
    )

    fig = go.Figure(data=[traceDirect, traceConnecting])
    fig.update_layout(
        xaxis_title='Value Bin',
        yaxis_title='Airline',
        template='plotly_white',
        width=width,
        height=height,
        updatemenus=[
            dict(
                active=0 if showDirect else 1,
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


# ----------- Heatmap Calls (only subheaders shown) -----------
st.subheader('Carbon Difference (Percentage)')
plotHeatmap(directFlights, connectingFlights, 'carbonDifferencePercent', colorscale='Reds')

st.subheader('Price (USD)')
plotHeatmap(directFlights, connectingFlights, 'price', colorscale='Reds')

st.subheader('Duration Time')
plotHeatmap(directFlights, connectingFlights, 'durationTime', colorscale='Reds')
