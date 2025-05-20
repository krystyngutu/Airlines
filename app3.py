import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ----------------------
# PAGE SETUP
# ----------------------
st.set_page_config(layout="wide")
st.title("Flights from NYC to CH")

# ----------------------
# HELPER FUNCTIONS
# ----------------------
def extract_parens_or_keep(val):
    """Extract text from parentheses or keep the original value."""
    if pd.isna(val):
        return val
    import re
    match = re.search(r'\((.*?)\)', val)
    return match.group(1) if match else val.strip()

def classify_aircraft(aircraft):
    """Standardize aircraft types into categories."""
    if pd.isna(aircraft):
        return "Other"
    aircraft = str(aircraft).lower()
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

def classify_flight_type(row, nyc_airports, swiss_airports):
    """Label flights as Direct or Connecting based on airports."""
    if row['departureAirportID'] in nyc_airports and row['arrivalAirportID'] in swiss_airports:
        return 'Direct'
    return 'Connecting'

# Create traces for time series plots
def create_traces(df):
    """Create traces for time series plots."""
    traces = []
    for airline in sorted(df['airline'].unique()):
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

# Bar chart helper with toggle for Direct vs Connecting
def plot_stacked_bars(direct_df, connecting_df, group_col, sub_col, legend_title, colors, show_direct, show_connecting):
    """Create stacked bar charts with toggles between direct and connecting flights."""
    def build_count(df):
        # Handle empty dataframes
        if df.empty:
            return pd.DataFrame()
            
        if not pd.api.types.is_categorical_dtype(df[sub_col]):
            df[sub_col] = pd.Categorical(df[sub_col])  # Ensure consistency
        counts = df.groupby([group_col, sub_col]).size().unstack(fill_value=0)

        for cat in df[sub_col].cat.categories:
            if cat not in counts.columns:
                counts[cat] = 0

        counts = counts.reindex(sorted(counts.columns), axis=1)
        return counts

    direct_count = build_count(direct_df)
    connecting_count = build_count(connecting_df)

    fig = go.Figure()
    direct_traces = []
    connecting_traces = []

    # Add direct traces
    if not direct_count.empty:
        for i, sub_category in enumerate(direct_count.columns):
            trace = go.Bar(
                x=direct_count.index,
                y=direct_count[sub_category],
                name=f'{sub_category}',
                marker_color=colors[i % len(colors)],
                visible=show_direct,
                legendgroup=f'{sub_category}',
                showlegend=True
            )
            fig.add_trace(trace)
            direct_traces.append(True)
            connecting_traces.append(False)
    
    # Add connecting traces
    if not connecting_count.empty:
        for i, sub_category in enumerate(connecting_count.columns):
            trace = go.Bar(
                x=connecting_count.index,
                y=connecting_count[sub_category],
                name=f'{sub_category}',
                marker_color=colors[i % len(colors)],
                visible=show_connecting,
                legendgroup=f'{sub_category}',
                showlegend=True
            )
            fig.add_trace(trace)
            direct_traces.append(False)
            connecting_traces.append(True)

    # If no traces were added, add a default "no data" message
    if len(direct_traces) == 0 and len(connecting_traces) == 0:
        fig.add_annotation(
            text="No data available for the selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )

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
                active=0 if show_direct else 1,
                buttons=[
                    dict(label="Direct Flights", method="update", args=[{"visible": direct_traces or [False]}]),
                    dict(label="Connecting Flights", method="update", args=[{"visible": connecting_traces or [False]}])
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

# Bubble chart helper function with flight type toggle
def plot_bubble_chart(direct_df, connecting_df, airline_col, metric_col, yaxis_title, show_direct, show_connecting, width=800, height=500):
    """Create bubble charts with toggles between direct and connecting flights."""
    def build_bubble(df):
        if df.empty:
            return pd.DataFrame()
            
        df['airline'] = pd.Categorical(df['airline'], categories=sorted(df['airline'].unique()), ordered=True)
        count_df = df.groupby(['airline', metric_col]).size().reset_index(name='count')
        count_df = count_df.sort_values('airline')  # Alphabetical order
        return count_df

    direct_data = build_bubble(direct_df)
    connecting_data = build_bubble(connecting_df)

    fig = go.Figure()
    
    # Add direct trace if data exists
    if not direct_data.empty:
        trace_direct = go.Scatter(
            x=direct_data[airline_col],
            y=direct_data[metric_col],
            mode='markers',
            text=direct_data['count'],
            marker=dict(
                size=direct_data['count'],
                color=direct_data[metric_col],
                colorscale='RdBu',
                showscale=True,
                sizemode='area',
                sizeref=2. * direct_data['count'].max() / (100 ** 2) if not direct_data.empty else 1,
                sizemin=4
            ),
            visible=show_direct
        )
        fig.add_trace(trace_direct)
    
    # Add connecting trace if data exists
    if not connecting_data.empty:
        trace_connecting = go.Scatter(
            x=connecting_data[airline_col],
            y=connecting_data[metric_col],
            mode='markers',
            text=connecting_data['count'],
            marker=dict(
                size=connecting_data['count'],
                color=connecting_data[metric_col],
                colorscale='RdBu',
                showscale=True,
                sizemode='area',
                sizeref=2. * connecting_data['count'].max() / (100 ** 2) if not connecting_data.empty else 1,
                sizemin=4
            ),
            visible=show_connecting
        )
        fig.add_trace(trace_connecting)
    
    # If no data available for either type, add a message
    if direct_data.empty and connecting_data.empty:
        fig.add_annotation(
            text="No data available for the selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )

    fig.update_layout(
        xaxis_title='Airline',
        yaxis_title=yaxis_title,
        template='plotly_white',
        showlegend=False,
        width=width,
        height=height,
        updatemenus=[
            dict(
                active=0 if show_direct else 1,
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

# Heatmap helper function with flight type toggle
def plot_heatmap(direct_df, connecting_df, value_col, show_direct, show_connecting, colorscale='Blues', width=800, height=500):
    """Create heatmaps with toggles between direct and connecting flights."""
    def build_heatmap_data(df):
        df_clean = df[[value_col, 'airline']].dropna()
        if df_clean.empty:
            return pd.DataFrame()

        # Sort airline names alphabetically
        df_clean['airline'] = df_clean['airline'].astype(str)
        airline_order = sorted(df_clean['airline'].unique())
        df_clean['airline'] = pd.Categorical(df_clean['airline'], categories=airline_order, ordered=True)

        binned_col = pd.cut(df_clean[value_col], bins=10)
        pivot = df_clean.groupby(['airline', binned_col]).size().unstack(fill_value=0)
        pivot = pivot.sort_index(level=0)
        return pivot

    direct_data = build_heatmap_data(direct_df)
    connecting_data = build_heatmap_data(connecting_df)

    fig = go.Figure()
    
    # Add direct trace if data exists
    if not direct_data.empty:
        trace_direct = go.Heatmap(
            z=direct_data.values,
            x=[str(col) for col in direct_data.columns],
            y=direct_data.index.tolist(),
            colorscale=colorscale,
            colorbar=dict(title='Number of Flights'),
            visible=show_direct
        )
        fig.add_trace(trace_direct)
    
    # Add connecting trace if data exists
    if not connecting_data.empty:
        trace_connecting = go.Heatmap(
            z=connecting_data.values,
            x=[str(col) for col in connecting_data.columns],
            y=connecting_data.index.tolist(),
            colorscale=colorscale,
            colorbar=dict(title='Number of Flights'),
            visible=show_connecting
        )
        fig.add_trace(trace_connecting)
    
    # Add annotation if no data
    if direct_data.empty and connecting_data.empty:
        fig.add_annotation(
            text="No data available for the selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )

    fig.update_layout(
        xaxis_title='Value Bin',
        yaxis_title='Airline',
        template='plotly_white',
        width=width,
        height=height,
        updatemenus=[
            dict(
                active=0 if show_direct else 1,
                buttons=[
                    dict(label="Direct Flights", method="update", args=[{"visible": [True, False] if not direct_data.empty and not connecting_data.empty else [True]}]),
                    dict(label="Connecting Flights", method="update", args=[{"visible": [False, True] if not direct_data.empty and not connecting_data.empty else [True]}])
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

# Cache data loading for better performance
@st.cache_data
def load_data():
    """Load and preprocess flight data."""
    try:
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
            split_ext = df['extensions'].str.split(',', n=2, expand=True).apply(lambda col: col.str.strip())
            df['recliningAndLegroom'] = split_ext[0]
            df['wifi'] = split_ext[1]
            df['carbonEmssionsEstimate'] = split_ext[2]

        df['recliningAndLegroom'] = df['recliningAndLegroom'].apply(extract_parens_or_keep)
        df['legroom'] = df['legroom'].fillna(df['recliningAndLegroom'])
        
        # Handle wifi column if it exists
        if 'wifi' in df.columns:
            df.loc[df['wifi'].str.startswith('Carbon', na=False), 'wifi'] = 'Status Unknown'

        # Derived features
        df['pricePerMinute'] = df['price'] / df['totalDurationMinutes']
        df['carbonDifferencePercent'] = (
            (df['carbonEmissionsThisFlight'] - df['carbonEmissionsThisFlight'].mean()) /
            df['carbonEmissionsThisFlight'].mean() * 100
        )
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return empty DataFrame if error

# ----------------------
# CONSTANTS
# ----------------------
# Define airline groups
direct_airlines = ['SWISS', 'United', 'Delta']
lufthansa_group = ['Austrian', 'Brussels Airlines', 'Discover Airlines', 'Eurowings', 'Edelweiss Air', 'ITA', 'Air Dolomiti', 'Lufthansa', 'SWISS']
star_alliance = ['Aegean', 'Air Canada', 'Air China', 'Air India', 'Air New Zealand', 'ANA', 'Asiana Airlines', 'Austrian', 'Avianca', 'Brussels Airlines', 'CopaAirlines', 'Croatia Airlines', 'Egyptair', 'Ethiopian Airlines', 'Eva Air', 'LOT Polish Airlines', 'Lufthansa', 'Shenzhen Airlines', 'Singapore Airlines', 'South African Airways', 'SWISS', 'Tap Air Portugal', 'Thai', 'Turkish Airlines', 'United']

# Define airports to include
nyc_airports = ["JFK", "EWR", "LGA"]
swiss_airports = ["ZRH", "GVA", "BSL"]

# Define airline colors
custom_colors = ['#d71920', '#00235f', '#f9ba00', '#660000', '#800080', '#3366ff',
                '#c3f550', '#fbaa3f', '#000000']

airline_colors = {
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
# DATA LOADING & FILTERING
# ----------------------
df = load_data()

if df.empty:
    st.error("Failed to load data. Please check your CSV file.")
    st.stop()

# Dropdown: default = Direct for general, Connecting for LHG and Star Alliance
filter_options = ['Airlines That Fly Both Direct and Connecting', 'Lufthansa Group', 'Star Alliance']
default_index = 0  # Default to general group

# Create session state if it doesn't exist
if 'filterChoice' not in st.session_state:
    st.session_state.filterChoice = filter_options[default_index]

filter_choice = st.selectbox("Select airlines to view:", options=filter_options, index=default_index, key='airline_filter')
st.session_state.filterChoice = filter_choice

# Set filtered airline list and default flight type
if filter_choice == 'Lufthansa Group':
    filtered_airlines = lufthansa_group
    show_direct = False
    show_connecting = True
elif filter_choice == 'Star Alliance':
    filtered_airlines = star_alliance
    show_direct = False
    show_connecting = True
else:
    filtered_airlines = direct_airlines
    show_direct = True
    show_connecting = False

# Filter DataFrame based on the selected airline group
try:
    df_filtered = df[df['airline'].isin(filtered_airlines)].copy()
except Exception as e:
    st.error(f"Error filtering data: {e}")
    st.stop()

# Label flights as Direct or Connecting
df_filtered['flightType'] = df_filtered.apply(
    lambda row: classify_flight_type(row, nyc_airports, swiss_airports), 
    axis=1
)

# Apply aircraft classification to all filtered data
df_filtered['airplaneLumped'] = df_filtered['airplane'].apply(classify_aircraft)

# Split into direct and connecting flights
direct_flights = df_filtered[df_filtered['flightType'] == 'Direct'].copy()
connecting_flights = df_filtered[df_filtered['flightType'] == 'Connecting'].copy()

# Drop rows with missing values for key comparisons
price_df = df_filtered.dropna(subset=['price', 'durationMinutes', 'carbonEmissionsThisFlight', 'legroom', 'travelClass', 'airplane'])

# ----------------------
# CHARTS ORGANIZED BY TOPIC
# ----------------------

# ===== 1. PRICE SECTION =====
st.header("1. Price Analysis")

# Price Over Time
st.subheader("Price Over Time")
direct_traces = create_traces(direct_flights)
connecting_traces = create_traces(connecting_flights)

fig = go.Figure()

# Add direct traces
for trace in direct_traces:
    trace.visible = show_direct
    fig.add_trace(trace)

# Add connecting traces
for trace in connecting_traces:
    trace.visible = show_connecting
    fig.add_trace(trace)

fig.update_layout(
    updatemenus=[
        dict(
            active=0 if show_direct else 1,
            buttons=[
                dict(label='Direct Flights',
                     method='update',
                     args=[{'visible':[True]*len(direct_traces) + [False]*len(connecting_traces)}]),
                dict(label='Connecting Flights',
                     method='update',
                     args=[{'visible':[False]*len(direct_traces) + [True]*len(connecting_traces)}])
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

st.plotly_chart(fig, use_container_width=True)

# Price vs Duration
st.subheader("Price vs Duration")
st.plotly_chart(go.Figure(
    data=[go.Scatter(
        x=price_df['totalDurationMinutes'],
        y=price_df['price'],
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

# Price by Travel Class
st.subheader("Price by Travel Class")
st.plotly_chart(go.Figure(
    data=[go.Box(
        x=price_df['travelClass'],
        y=price_df['price'],
        name='Travel Class',
        marker_color='purple'
    )],
    layout=go.Layout(
        xaxis_title="Travel Class",
        yaxis_title="Price (USD)",
        height=450
    )
), use_container_width=True)

# Price Heatmap
st.subheader('Price Distribution by Airline')
plot_heatmap(direct_flights, connecting_flights, 'price', show_direct, show_connecting, colorscale='Reds')

# ===== 2. LEGROOM SECTION =====
st.header("2. Legroom Analysis")

# Price vs Legroom
st.subheader("Price vs Legroom")

# Filter valid values
valid_legroom = df_filtered[df_filtered['legroom'].notna() & df_filtered['price'].notna()]
legroom_grouped = valid_legroom.groupby('legroom')['price'].mean().reset_index()
legroom_grouped = legroom_grouped.sort_values(by='legroom')

fig_legroom = go.Figure(go.Scatter(
    x=legroom_grouped['legroom'],
    y=legroom_grouped['price'],
    mode='lines+markers',
    line=dict(color='#d71920'),
    marker=dict(size=8)
))

fig_legroom.update_layout(
    xaxis_title='Legroom (inches or category)',
    yaxis_title='Average Price (USD)',
    template='plotly_white',
    height=450
)

st.plotly_chart(fig_legroom, use_container_width=True)

# Price by Legroom (box)
st.subheader("Price Distribution by Legroom")
st.plotly_chart(go.Figure(
    data=[go.Box(
        x=price_df['legroom'],
        y=price_df['price'],
        name='Legroom',
        marker_color='orange'
    )],
    layout=go.Layout(
        xaxis_title="Legroom (inches or category)",
        yaxis_title="Price (USD)",
        height=450
    )
), use_container_width=True)

# Legroom breakdown by airline
st.subheader('Legroom Options by Airline')
plot_stacked_bars(
    direct_flights,
    connecting_flights,
    group_col='airline',
    sub_col='legroom',
    legend_title='Legroom',
    colors=custom_colors,
    show_direct=show_direct,
    show_connecting=show_connecting
)

# ===== 3. AIRCRAFT SECTION =====
st.header("3. Aircraft Analysis")

# Price vs Aircraft (lumped groups)
st.subheader("Price by Aircraft Type")

# Filter and group
try:
    valid_aircraft = df_filtered[df_filtered['airplaneLumped'].notna() & df_filtered['price'].notna()]
    aircraft_grouped = valid_aircraft.groupby('airplaneLumped')['price'].mean().reset_index()

    fig_aircraft = go.Figure(go.Scatter(
        x=aircraft_grouped['airplaneLumped'],
        y=aircraft_grouped['price'],
        mode='lines+markers',
        line=dict(color='#00235f'),
        marker=dict(size=8)
    ))

    fig_aircraft.update_layout(
        xaxis_title='Aircraft Type',
        yaxis_title='Average Price (USD)',
        template='plotly_white',
        height=450
    )

    st.plotly_chart(fig_aircraft, use_container_width=True)
except Exception as e:
    st.error(f"Error creating aircraft price chart: {e}")

# Price by Aircraft (box)
st.subheader("Price Distribution by Aircraft Type")
st.plotly_chart(go.Figure(
    data=[go.Box(
        x=price_df['airplane'],
        y=price_df['price'],
        name='Aircraft',
        marker_color='darkred'
    )],
    layout=go.Layout(
        xaxis_title="Aircraft Type",
        yaxis_title="Price (USD)",
        height=450
    )
), use_container_width=True)

# Aircraft by Airline
st.subheader('Aircraft Types by Airline')
plot_stacked_bars(
    direct_flights,
    connecting_flights,
    group_col='airline',
    sub_col='airplaneLumped',
    legend_title='Aircraft',
    colors=custom_colors,
    show_direct=show_direct,
    show_connecting=show_connecting
)

# ===== 4. CARBON EMISSIONS SECTION =====
st.header("4. Carbon Emissions Analysis")

# Price vs Carbon Emissions
st.subheader("Price vs Carbon Emissions")

# Prepare traces for scatter plot
carbon_direct_traces = []
carbon_connecting_traces = []

for airline in sorted(df_filtered['airline'].unique()):
    # Direct flights
    data_direct = direct_flights[direct_flights['airline'] == airline]
    if not data_direct.empty:
        carbon_direct_traces.append(go.Scatter(
            x=data_direct['carbonEmissionsThisFlight'],
            y=data_direct['price'],
            mode='markers',
            name=airline,
            hovertext=data_direct['flightNumber'],
            marker=dict(color=airline_colors.get(airline, 'gray'))
        ))

    # Connecting flights
    data_connecting = connecting_flights[connecting_flights['airline'] == airline]
    if not data_connecting.empty:
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

# Add direct traces
for trace in carbon_direct_traces:
    trace.visible = show_direct
    carbon_fig.add_trace(trace)

# Add connecting traces
for trace in carbon_connecting_traces:
    trace.visible = show_connecting
    carbon_fig.add_trace(trace)

carbon_fig.update_layout(
    updatemenus=[
        dict(
            active=0 if show_direct else 1,
            buttons=[
                dict(label="Direct Flights",
                     method="update",
                     args=[{"visible": [True]*len(carbon_direct_traces) + [False]*len(carbon_connecting_traces)}]),
                dict(label="Connecting Flights",
                     method="update",
                     args=[{"visible": [False]*len(carbon_direct_traces) + [True]*len(carbon_connecting_traces)}])
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

st.plotly_chart(carbon_fig, use_container_width=True)

# Simple scatter for carbon emissions
st.subheader("Carbon Emissions vs Price (All Airlines)")
st.plotly_chart(go.Figure(
    data=[go.Scatter(
        x=price_df['carbonEmissionsThisFlight'],
        y=price_df['price'],
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

# Carbon emissions breakdown
st.subheader('Carbon Emissions by Airline')
plot_bubble_chart(
    direct_flights, 
    connecting_flights, 
    'airline', 
    'carbonEmissionsThisFlight', 
    'Carbon Emissions (kg CO₂)',
    show_direct,
    show_connecting
)

# Carbon difference breakdown
st.subheader('Carbon Difference (%) by Airline')
plot_bubble_chart(
    direct_flights, 
    connecting_flights, 
    'airline', 
    'carbonDifferencePercent', 
    'Carbon Difference (%)',
    show_direct,
    show_connecting
)

# Carbon Difference Heatmap
st.subheader('Carbon Emissions Distribution by Airline')
plot_heatmap(
    direct_flights, 
    connecting_flights, 
    'carbonDifferencePercent', 
    show_direct, 
    show_connecting,
    colorscale='Reds'
)

# ===== 5. DURATION SECTION =====
st.header("5. Duration Analysis")

# Flight duration breakdown
st.subheader('Flight Duration by Airline')
plot_bubble_chart(
    direct_flights, 
    connecting_flights, 
    'airline', 
    'durationMinutes', 
    'Duration (min)',
    show_direct,
    show_connecting
)

# Duration Heatmap
st.subheader('Flight Duration Distribution by Airline')
plot_heatmap(
    direct_flights, 
    connecting_flights, 
    'durationTime', 
    show_direct, 
    show_connecting,
    colorscale='Blues'
)

# ===== 6. AMENITIES SECTION =====
st.header("6. Amenities Analysis")

# Get all unique wifi categories
if 'wifi' in df_filtered.columns:
    wifi_categories = sorted(set(direct_flights['wifi'].dropna().unique()).union(connecting_flights['wifi'].dropna().unique()))

    # Convert to ordered categorical
    if not direct_flights.empty and 'wifi' in direct_flights.columns:
        direct_flights['wifi'] = pd.Categorical(direct_flights['wifi'], categories=wifi_categories, ordered=True)
    if not connecting_flights.empty and 'wifi' in connecting_flights.columns:
        connecting_flights['wifi'] = pd.Categorical(connecting_flights['wifi'], categories=wifi_categories, ordered=True)

    # WiFi breakdown
    st.subheader('WiFi Options by Airline')
    plot_stacked_bars(
        direct_flights,
        connecting_flights,
        group_col='airline',
        sub_col='wifi',
        legend_title='WiFi',
        colors=custom_colors,
        show_direct=show_direct,
        show_connecting=show_connecting
    )

# Travel Class breakdown
st.subheader('Travel Classes by Airline')
plot_stacked_bars(
    direct_flights,
    connecting_flights,
    group_col='airline',
    sub_col='travelClass',
    legend_title='Travel Class',
    colors=custom_colors,
    show_direct=show_direct,
    show_connecting=show_connecting
)
