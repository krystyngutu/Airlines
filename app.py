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
lufthansaGroup = ['Austrian', 'Brussels Airlines', 'Discover Airlines', 'Eurowings', 'Edelweiss Air', 'ITA', 'Air Dolomiti', 'Lufthansa', 'SWISS']
starAlliance = ['Aegean', 'Air Canada', 'Air China', 'Air India', 'Air New Zealand', 'ANA', 'Asiana Airlines', 'Austrian', 'Avianca', 'Brussels Airport', 'CopaAirlines', 'Croatia Airlines', 'Egyptair', 'Ethiopian Airlines', 'Eva Air', 'LOT Polish Airlines', 'Lufthansa', 'Shenzhen Airlines', 'Singapore Airlines', 'South African Airways', 'SWISS', 'Tap Air Portugal', 'Thai', 'Turkish Airlines', 'United']

# Toggle for connected flights
showConnected = st.toggle("Include All Airlines", value=False)

# Filtering options
if not showConnected:
    # Default: show only direct airlines
    filteredAirlines = directAirlines
else:
    # User selects airline group when showing connecting flights
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
    trace.visible=True
    fig.add_trace(trace)

# Add connecting traces (hidden)
for trace in connectingTraces:
    trace.visible=False
    fig.add_trace(trace)

fig.update_layout(
    updatemenus=[
        dict(
            active=0,
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
    legend_title_text="Airlines",
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
    legend_title_text="Airlines",
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
    def buildCount(df):
        if not pd.api.types.is_categorical_dtype(df[sub_col]):
            df[sub_col] = pd.Categorical(df[sub_col])  # Ensure consistency
        counts = df.groupby([group_col, sub_col]).size().unstack(fill_value=0)

        # Ensure all subcategories are included
        for cat in df[sub_col].cat.categories:
            if cat not in counts.columns:
                counts[cat] = 0
        
        counts = counts[sorted(counts.columns)]
        return counts

    directCount = buildCount(directDF)
    connectingCount = buildCount(connectingDF)

    fig = go.Figure()

    directTraces = []
    connectingTraces = []

    for i, sub_category in enumerate(directCount.columns):
        fig.add_trace(go.Bar(
            x=directCount.index,
            y=directCount[sub_category],
            name=sub_category,
            marker_color=colors[i % len(colors)],
            visible=True
        ))
        directTraces.append(True)
        connectingTraces.append(False)

    for i, sub_category in enumerate(connectingCount.columns):
        fig.add_trace(go.Bar(
            x=connectingCount.index,
            y=connectingCount[sub_category],
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
        updatemenus=[
            dict(
                active=0,
                buttons=[
                    dict(label="Direct Flights",
                         method="update",
                         args=[{"visible": directTraces}]),
                    dict(label="Connecting Flights",
                         method="update",
                         args=[{"visible": connectingTraces}])
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

# Legroom breakdown
st.subheader('Legroom by Airline')
plotlyStackedBars(
    directFlights,
    connectingFlights,
    group_col='airline',
    sub_col='legroom',
    legend_title='Legroom',
    colors=customColors
)

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

# Bubble chart helper function with flight type toggle
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
        mode='markers+text' if metric_col == 'durationTime' else 'markers',
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
        mode='markers+text' if metric_col == 'durationTime' else 'markers',
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
st.subheader('Flight Duration by Airline (Bubble Size = Count)')
plotBubbleChart(directFlights, connectingFlights, 'airline', 'durationTime',
                'Duration (min)', width=1000)

# Flight prices breakdown
st.subheader('Flight Prices by Airline (Bubble Size = Count)')
plotBubbleChart(directFlights, connectingFlights, 'airline', 'price', 'Price (USD)')

# Carbon emissions breakdown
st.subheader('Carbon Emissions by Airline per Flight (Bubble Size = Count)')
plotBubbleChart(directFlights, connectingFlights, 'airline', 'carbonEmissionsThisFlight',
                'Carbon Emissions by Airline (Bubble Size = Count)')

# Carbon difference breakdown
st.subheader('Carbon Difference (%) by Airline per Flight (Bubble Size = Count)')
plotBubbleChart(directFlights, connectingFlights, 'airline', 'carbonDifferencePercent',
                'Carbon Difference by Airline (Bubble Size = Count)')

# Heatmap helper function with flight type toggle
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
        title=title,
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
                               {"title": title + " (Direct)"}]),
                    dict(label="Connecting Flights",
                         method="update",
                         args=[{"visible": [False, True]},
                               {"title": title + " (Connecting)"}])
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

# Heatmaps
# Carbon Difference Percent by Airline
plotHeatmap(directFlights, connectingFlights, 'carbonDifferencePercent',
           'Carbon Difference Percent', colorscale='Reds')

# Price by Airline
plotHeatmap(directFlights, connectingFlights, 'price', 'Price (USD)', colorscale='Reds')

# Duration Time by Airline
plotHeatmap(directFlights, connectingFlights, 'durationTime', 'Duration (min)', colorscale='Reds')
