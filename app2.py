import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page configuration
st.set_page_config(
    page_title="Flight Booking Optimizer",
    page_icon="✈️",
    layout="wide"
)

# App title and description
st.title("✈️ Flight Booking Optimizer")
st.markdown("""
This app analyzes flight data to help you decide **when to book** and **which airline to choose** 
based on price, carbon emissions, and other factors.
""")

# Function to load and preprocess data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Convert departure and arrival times to datetime
    df['departureTime'] = pd.to_datetime(df['departureTime'], errors='coerce')
    df['arrivalAirportTime'] = pd.to_datetime(df['arrivalAirportTime'], errors='coerce')
    
    # Extract useful features
    df['bookingDay'] = df['departureTime'].dt.day_name()
    df['bookingMonth'] = df['departureTime'].dt.month_name()
    df['bookingHour'] = df['departureTime'].dt.hour
    df['timeOfDay'] = pd.cut(
        df['bookingHour'], 
        bins=[0, 5, 12, 17, 21, 24], 
        labels=['Late Night', 'Morning', 'Afternoon', 'Evening', 'Night']
    )
    
    # Calculate days until departure (from current date)
    today = datetime.now()
    df['daysUntilDeparture'] = (df['departureTime'] - today).dt.days
    df['bookingLeadTime'] = pd.cut(
        df['daysUntilDeparture'],
        bins=[-float('inf'), 7, 14, 30, float('inf')],
        labels=['0-7 days', '8-14 days', '15-30 days', '30+ days']
    )
    
    # Clean up data
    df = df.dropna(subset=['price', 'departureTime', 'airline'])
    
    return df

# Sidebar for uploading file or using example data
st.sidebar.header("Data Input")

# Option to use example data or upload custom data
data_option = st.sidebar.radio(
    "Choose data source:",
    ["Use uploaded CSV file", "Use example data (built-in)"]
)

if data_option == "Use uploaded CSV file":
    uploaded_file = st.sidebar.file_uploader("Upload your flight data CSV", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.sidebar.success("✅ Data successfully loaded!")
    else:
        st.sidebar.info("Please upload a CSV file to continue")
        # Create sample data if no file is uploaded
        sample_data = {
            'departureAirportName': ['New York', 'London', 'Paris'] * 5,
            'arrivalAirportName': ['London', 'Paris', 'New York'] * 5,
            'departureTime': [(datetime.now() + timedelta(days=i)) for i in range(15)],
            'arrivalAirportTime': [(datetime.now() + timedelta(days=i, hours=8)) for i in range(15)],
            'airline': ['United', 'British Airways', 'Air France', 'Lufthansa', 'Delta'] * 3,
            'price': [900, 1200, 1000, 850, 1100] * 3,
            'durationTime': [480, 520, 460, 500, 490] * 3,
            'carbonEmissionsThisFlight': [450000, 520000, 480000, 460000, 500000] * 3,
            'carbonEmissionsTypicalRoute': [470000, 500000, 490000, 470000, 480000] * 3,
            'travelClass': ['Economy', 'Economy', 'Business', 'Economy', 'Premium Economy'] * 3
        }
        df = pd.DataFrame(sample_data)
        df = load_data(pd.DataFrame(sample_data).to_csv('temp.csv', index=False))
else:
    # Use the provided flight data
    df = load_data("all_flights.csv")
    st.sidebar.success("✅ Example data loaded!")

# Sidebar for model parameters
st.sidebar.header("Optimization Parameters")

price_weight = st.sidebar.slider(
    "Price importance", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.7,
    help="Higher value means price is more important in your decision"
)

emission_weight = st.sidebar.slider(
    "Carbon emissions importance", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.3,
    help="Higher value means emissions are more important in your decision"
)

st.sidebar.header("Filter Options")
min_price = int(df['price'].min())
max_price = int(df['price'].max())
price_range = st.sidebar.slider(
    "Price Range ($)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price)
)

airlines = st.sidebar.multiselect(
    "Airlines",
    options=sorted(df['airline'].unique()),
    default=sorted(df['airline'].unique())[:5]
)

# Apply filters
filtered_df = df[
    (df['price'] >= price_range[0]) & 
    (df['price'] <= price_range[1]) &
    (df['airline'].isin(airlines))
]

# Main content
if len(filtered_df) > 0:
    # Key metrics in 4 columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cheapest_airline = df.groupby('airline')['price'].mean().idxmin()
        cheapest_price = int(df.groupby('airline')['price'].mean().min())
        st.metric("Cheapest Airline", cheapest_airline, f"${cheapest_price}")
    
    with col2:
        best_day = df.groupby('bookingDay')['price'].mean().idxmin()
        day_price = int(df.groupby('bookingDay')['price'].mean().min())
        st.metric("Best Day to Fly", best_day, f"${day_price}")
    
    with col3:
        best_time = df.groupby('timeOfDay')['price'].mean().idxmin()
        time_price = int(df.groupby('timeOfDay')['price'].mean().min())
        st.metric("Best Time to Fly", best_time, f"${time_price}")
    
    with col4:
        greenest_airline = df.groupby('airline')['carbonEmissionsThisFlight'].mean().idxmin()
        green_emissions = int(df.groupby('airline')['carbonEmissionsThisFlight'].mean().min() / 1000)
        st.metric("Greenest Airline", greenest_airline, f"{green_emissions} kg CO₂")
    
    st.markdown("---")
    
    # Data overview
    with st.expander("📊 Data Overview"):
        st.write(f"Total flights: {len(filtered_df)}")
        st.dataframe(filtered_df.head(10))
        
        # Distribution of prices
        fig = px.histogram(
            filtered_df, 
            x="price", 
            color="airline",
            title="Distribution of Flight Prices",
            labels={"price": "Price ($)", "count": "Number of Flights"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏢 Airline Analysis", 
        "📅 Timing Analysis", 
        "🌿 Emissions Analysis",
        "🔮 Price Prediction"
    ])
    
    with tab1:
        st.header("Airline Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average price by airline
            airline_prices = filtered_df.groupby('airline')['price'].mean().sort_values()
            
            fig = px.bar(
                airline_prices.reset_index(), 
                x='airline', 
                y='price',
                title="Average Price by Airline",
                labels={"price": "Average Price ($)", "airline": "Airline"},
                color='price',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Price vs Duration scatterplot
            fig = px.scatter(
                filtered_df,
                x="durationTime",
                y="price",
                color="airline",
                size="price",
                title="Price vs. Flight Duration by Airline",
                labels={
                    "durationTime": "Flight Duration (minutes)",
                    "price": "Price ($)",
                    "airline": "Airline"
                },
                hover_data=["departureAirportName", "arrivalAirportName", "travelClass"]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Calculate airline value score (weighted average of price and emissions)
        # Lower score is better (lower price, lower emissions)
        airline_data = filtered_df.groupby('airline').agg({
            'price': 'mean',
            'carbonEmissionsThisFlight': 'mean',
            'durationTime': 'mean'
        }).reset_index()
        
        # Scale the values for fair comparison
        scaler = StandardScaler()
        airline_data[['price_scaled', 'emissions_scaled']] = scaler.fit_transform(
            airline_data[['price', 'carbonEmissionsThisFlight']]
        )
        
        # Calculate the weighted score
        airline_data['value_score'] = (
            airline_data['price_scaled'] * price_weight + 
            airline_data['emissions_scaled'] * emission_weight
        )
        
        # Sort by the value score (lower is better)
        airline_data = airline_data.sort_values('value_score')
        
        st.subheader("Best Overall Airlines (based on your priorities)")
        st.write("Lower score indicates better value based on your price and emissions preferences")
        
        # Create color mapping (green for good, red for bad)
        norm = plt.Normalize(airline_data['value_score'].min(), airline_data['value_score'].max())
        colors = plt.cm.RdYlGn_r(norm(airline_data['value_score']))
        
        fig = go.Figure(data=[
            go.Bar(
                x=airline_data['airline'],
                y=airline_data['value_score'],
                marker_color=colors.tolist(),
                text=round(airline_data['value_score'], 2),
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Airline Value Score (Lower is Better)",
            xaxis_tickangle=-45,
            yaxis_title="Score (Price & Emissions Weighted)",
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the top 5 airlines in a table
        top5_airlines = airline_data.head(5)
        top5_airlines['price'] = top5_airlines['price'].round(0).astype(int)
        top5_airlines['carbon_kg'] = (top5_airlines['carbonEmissionsThisFlight'] / 1000).round(0).astype(int)
        top5_airlines['duration_hrs'] = (top5_airlines['durationTime'] / 60).round(1)
        
        st.dataframe(
            top5_airlines[['airline', 'price', 'carbon_kg', 'duration_hrs']].rename(
                columns={
                    'airline': 'Airline',
                    'price': 'Avg Price ($)',
                    'carbon_kg': 'Carbon (kg)',
                    'duration_hrs': 'Duration (hrs)'
                }
            )
        )
    
    with tab2:
        st.header("Timing Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Day of week analysis
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Make sure we have all days of the week
            day_prices = {day: filtered_df[filtered_df['bookingDay'] == day]['price'].mean() for day in day_order if day in filtered_df['bookingDay'].unique()}
            day_prices = pd.Series(day_prices)
            
            fig = px.bar(
                x=day_prices.index, 
                y=day_prices.values,
                title="Average Price by Day of Week",
                labels={"x": "Day of Week", "y": "Average Price ($)"},
                color=day_prices.values,
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig.update_layout(xaxis_tickangle=0)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Time of day analysis
            time_order = ['Late Night', 'Morning', 'Afternoon', 'Evening', 'Night']
            
            # Make sure we have all time categories
            time_prices = {time: filtered_df[filtered_df['timeOfDay'] == time]['price'].mean() for time in time_order if time in filtered_df['timeOfDay'].unique()}
            time_prices = pd.Series(time_prices)
            
            fig = px.bar(
                x=time_prices.index, 
                y=time_prices.values,
                title="Average Price by Time of Day",
                labels={"x": "Time of Day", "y": "Average Price ($)"},
                color=time_prices.values,
                color_continuous_scale=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Lead time analysis
        lead_time_order = ['0-7 days', '8-14 days', '15-30 days', '30+ days']
        
        # Make sure we have all lead time categories
        lead_time_prices = {lt: filtered_df[filtered_df['bookingLeadTime'] == lt]['price'].mean() for lt in lead_time_order if lt in filtered_df['bookingLeadTime'].unique()}
        lead_time_prices = pd.Series(lead_time_prices)
        
        fig = px.bar(
            x=lead_time_prices.index, 
            y=lead_time_prices.values,
            title="Average Price by Booking Lead Time",
            labels={"x": "Days Before Departure", "y": "Average Price ($)"},
            color=lead_time_prices.values,
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Combined analysis of day of week and time of day
        if 'bookingDay' in filtered_df.columns and 'timeOfDay' in filtered_df.columns:
            heat_data = filtered_df.pivot_table(
                index='bookingDay', 
                columns='timeOfDay', 
                values='price', 
                aggfunc='mean'
            )
            
            # Reindex to ensure the days are in the correct order
            if set(day_order).issubset(set(heat_data.index)):
                heat_data = heat_data.reindex(day_order)
            
            # Reindex to ensure times are in the correct order
            if set(time_order).issubset(set(heat_data.columns)):
                heat_data = heat_data[time_order]
            
            fig = px.imshow(
                heat_data,
                labels=dict(x="Time of Day", y="Day of Week", color="Price ($)"),
                x=heat_data.columns,
                y=heat_data.index,
                color_continuous_scale=px.colors.sequential.Viridis,
                title="Average Price by Day and Time"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            ### Key Takeaways for Timing:
            - **Best lead time**: Book either very close to departure (0-7 days) or well in advance (30+ days)
            - **Best days**: Wednesday to Friday typically have lower prices
            - **Best time**: Late night and early morning flights tend to be cheaper
            """)
    
    with tab3:
        st.header("Emissions Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average emissions by airline
            airline_emissions = filtered_df.groupby('airline')['carbonEmissionsThisFlight'].mean().sort_values() / 1000  # Convert to kg
            
            fig = px.bar(
                airline_emissions.reset_index(), 
                x='airline', 
                y='carbonEmissionsThisFlight',
                title="Average Carbon Emissions by Airline (kg)",
                labels={"carbonEmissionsThisFlight": "Average Emissions (kg)", "airline": "Airline"},
                color='carbonEmissionsThisFlight',
                color_continuous_scale=px.colors.sequential.Viridis_r  # Reversed so green is low emissions
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Price vs Emissions scatterplot
            fig = px.scatter(
                filtered_df,
                x="price",
                y="carbonEmissionsThisFlight",
                color="airline",
                size="durationTime",
                title="Price vs. Carbon Emissions by Airline",
                labels={
                    "price": "Price ($)",
                    "carbonEmissionsThisFlight": "Carbon Emissions (g)",
                    "airline": "Airline"
                },
                hover_data=["departureAirportName", "arrivalAirportName"]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Calculate carbon efficiency (emissions per dollar)
        filtered_df['carbon_per_dollar'] = filtered_df['carbonEmissionsThisFlight'] / filtered_df['price']
        airline_efficiency = filtered_df.groupby('airline')['carbon_per_dollar'].mean().sort_values() / 1000  # Convert to kg per dollar
        
        fig = px.bar(
            airline_efficiency.reset_index(), 
            x='airline', 
            y='carbon_per_dollar',
            title="Carbon Efficiency by Airline (kg CO₂ per dollar)",
            labels={"carbon_per_dollar": "kg CO₂ per $", "airline": "Airline"},
            color='carbon_per_dollar',
            color_continuous_scale=px.colors.sequential.Viridis_r  # Reversed so green is low emissions
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Compare emissions to typical route
        if 'carbonDifferencePercent' in filtered_df.columns:
            airline_diff = filtered_df.groupby('airline')['carbonDifferencePercent'].mean().sort_values()
            
            fig = px.bar(
                airline_diff.reset_index(),
                x='airline',
                y='carbonDifferencePercent',
                title="Carbon Emissions vs. Typical Route (%)",
                labels={"carbonDifferencePercent": "Difference (%)", "airline": "Airline"},
                color='carbonDifferencePercent',
                color_continuous_scale=px.colors.sequential.Viridis_r
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            ### Key Takeaways for Emissions:
            - Airlines with newer fleets generally have lower emissions
            - Shorter flights and non-stop routes typically have lower total emissions
            - Price and emissions don't always correlate - you can often find affordable options with lower emissions
            """)
    
    with tab4:
        st.header("Price Prediction Model")
        
        # Create features for the model
        # Base features
        st.write("Building a price prediction model based on flight characteristics...")
        
        # Select features for training
        X_features = filtered_df[['durationTime']].copy()
        
        # Add one-hot encoded features
        if 'airline' in filtered_df.columns:
            airline_dummies = pd.get_dummies(filtered_df['airline'], prefix='airline')
            X_features = pd.concat([X_features, airline_dummies], axis=1)
        
        if 'bookingDay' in filtered_df.columns:
            day_dummies = pd.get_dummies(filtered_df['bookingDay'], prefix='day')
            X_features = pd.concat([X_features, day_dummies], axis=1)
        
        if 'timeOfDay' in filtered_df.columns:
            time_dummies = pd.get_dummies(filtered_df['timeOfDay'], prefix='time')
            X_features = pd.concat([X_features, time_dummies], axis=1)
        
        if 'bookingLeadTime' in filtered_df.columns:
            leadtime_dummies = pd.get_dummies(filtered_df['bookingLeadTime'], prefix='leadtime')
            X_features = pd.concat([X_features, leadtime_dummies], axis=1)
        
        # Target variable
        y = filtered_df['price']
        
        # Train model if we have enough data
        if len(X_features) >= 100:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, y, test_size=0.2, random_state=42
            )
            
            # Create and train the model
            model = GradientBoostingRegressor(random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, train_preds)
            test_mae = mean_absolute_error(y_test, test_preds)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            
            train_r2 = r2_score(y_train, train_preds)
            test_r2 = r2_score(y_test, test_preds)
            
            # Show model performance
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Test MAE", f"${int(test_mae)}")
                st.caption("Mean Absolute Error (lower is better)")
            
            with col2:
                st.metric("Test RMSE", f"${int(test_rmse)}")
                st.caption("Root Mean Squared Error (lower is better)")
            
            with col3:
                st.metric("Test R²", f"{test_r2:.2f}")
                st.caption("Coefficient of Determination (higher is better)")
            
            # Feature importance
            features = X_features.columns
            importances = model.feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10 features
            
            fig = px.bar(
                x=importances[indices],
                y=[features[i] for i in indices],
                orientation='h',
                title="Top 10 Features Influencing Price",
                labels={"x": "Importance", "y": "Feature"}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Price prediction tool
            st.subheader("Price Prediction Tool")
            st.write("Use this tool to predict flight prices based on your preferences:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pred_airline = st.selectbox("Airline", options=sorted(filtered_df['airline'].unique()))
                pred_day = st.selectbox("Day of Week", options=day_order)
            
            with col2:
                pred_time = st.selectbox("Time of Day", options=time_order)
                pred_leadtime = st.selectbox("Booking Lead Time", options=lead_time_order)
            
            with col3:
                avg_duration = int(filtered_df['durationTime'].mean())
                pred_duration = st.slider("Flight Duration (minutes)", 
                                         min_value=int(filtered_df['durationTime'].min()),
                                         max_value=int(filtered_df['durationTime'].max()),
                                         value=avg_duration)
            
            # Create prediction input
            pred_input = pd.DataFrame({
                'durationTime': [pred_duration]
            })
            
            # Add categorical variables
            for airline in airline_dummies.columns:
                pred_input[airline] = 1 if airline == f'airline_{pred_airline}' else 0
            
            for day in day_dummies.columns:
                pred_input[day] = 1 if day == f'day_{pred_day}' else 0
            
            for time in time_dummies.columns:
                pred_input[time] = 1 if time == f'time_{pred_time}' else 0
            
            for leadtime in leadtime_dummies.columns:
                pred_input[leadtime] = 1 if leadtime == f'leadtime_{pred_leadtime}' else 0
            
            # Make sure all columns match the training data
            missing_cols = set(X_features.columns) - set(pred_input.columns)
            for col in missing_cols:
                pred_input[col] = 0
                
            pred_input = pred_input[X_features.columns]
            
            # Make prediction
            predicted_price = model.predict(pred_input)[0]
            
            # Display prediction
            st.markdown("### Predicted Price")
            st.markdown(f"<h1 style='text-align: center; color: #1E88E5;'>${int(predicted_price)}</h1>", unsafe_allow_html=True)
            
            # Compare to average
            avg_price = filtered_df[filtered_df['airline'] == pred_airline]['price'].mean()
            diff_from_avg = predicted_price - avg_price
            diff_percent = (diff_from_avg / avg_price) * 100
            
            if diff_from_avg < 0:
                st.success(f"This is ${abs(int(diff_from_avg))} ({abs(diff_percent):.1f}%) below the average price for {pred_airline}")
            else:
                st.warning(f"This is ${int(diff_from_avg)} ({diff_percent:.1f}%) above the average price for {pred_airline}")
        
        else:
            st.warning("Not enough data to build a reliable prediction model. Please upload more flight data or adjust your filters.")
    
    # Final recommendations section
    st.markdown("---")
    st.header("🎯 Your Personalized Recommendations")
    
    # Calculate optimized recommendations
    if price_weight >= emission_weight:
        st.write("Based on your priorities with price being more important:")
        
        # Best airline by price
        best_price_airline = filtered_df.groupby('airline')['price'].mean().idxmin()
        best_price = int(filtered_df.groupby('airline')['price'].mean().min())
        
        # Best day by price
        best_price_day = filtered_df.groupby('bookingDay')['price'].mean().idxmin() if 'bookingDay' in filtered_df.columns else "N/A"
        
        # Best lead time by price
        best_price_leadtime = filtered_df.groupby('bookingLeadTime')['price'].mean().idxmin() if 'bookingLeadTime' in filtered_df.columns else "N/A"
        
        st.info(f"""
        1. **Airline Recommendation**: Book with **{best_price_airline}** for the best prices (avg. ${best_price})
        2. **When to Fly**: If possible, fly on **{best_price_day}** during **{best_time}** hours
        3. **When to Book**: Book **{best_price_leadtime}** before your travel date
        """)
    else:
        st.write("Based on your priorities with emissions being more important:")
        
        # Best airline by emissions
        best_emissions_airline = filtered_df.groupby('airline')['carbonEmissionsThisFlight'].mean().idxmin()
        best_emissions = int(filtered_df.groupby('airline')['carbonEmissionsThisFlight'].mean().min() / 1000)
        
        # Most efficient airline
        efficiency_data = filtered_df.groupby('airline').apply(
            lambda x: (x['carbonEmissionsThisFlight'] / x['price']).mean()
        ).sort_values()
        most_efficient_airline = efficiency_data.index[0]
        
        st.info(f"""
        1. **Airline Recommendation**: Book with **{best_emissions_airline}** for the lowest emissions (avg. {best_emissions} kg CO₂)
        2. **Best Value Option**: For balancing cost and emissions, consider **{most_efficient_airline}**
        3. **When to Book**: Book **{best_price_leadtime}** before your travel date, as this doesn't significantly impact emissions
        """)
    
    # Planning strategies
    st.markdown("""
    ### 📈 Strategic Planning Tips
    
    1. **Price Volatility Strategy**: Prices tend to be most volatile 8-14 days before departure. If booking during this window, consider whether waiting or booking immediately would be more advantageous.
    
    2. **Carbon-Conscious Booking**: Airlines with newer fleets typically have lower emissions. When comparing similar prices, check the aircraft type - newer models are generally more fuel-efficient.
    
    3. **Balanced Approach**: For the best overall value, focus on airlines that scored well in the "Airline Value Score" chart, as these provide the best combination of price and emissions based on your preferences.
    
    4. **Seasonal Considerations**: If your travel dates are flexible, consider how seasonality affects your route. Shoulder seasons often provide the best balance of price and experience.
    """)
    
    # Advanced models
    st.header("🧠 Advanced Predictive Models")
    
    st.markdown("""
    Based on the data, I recommend building these predictive models to optimize your flight booking decisions:
    
    1. **Price Prediction Model**
    
       This model is already implemented above in the "Price Prediction" tab. It uses features like:
       - Airline
       - Day of week
       - Time of day
       - Booking lead time
       - Flight duration
       
       The model helps you estimate prices for different combinations of these factors, allowing you to optimize your booking decisions.
    
    2. **Price Volatility Model**
    
       This would predict how prices might change as you get closer to the departure date. It would help you decide whether to:
       - Book now
       - Wait for potential price drops
       - Set up price alerts at specific thresholds
       
       Key features for this model:
       - Historical price trends by route
       - Days until departure
       - Seasonality factors
       - Booking demand indicators
    
    3. **Carbon Emissions Estimator**
    
       This model would predict carbon emissions for flights not in your dataset. Useful for making environmentally-conscious booking decisions. Key features:
       - Aircraft type
       - Flight distance
       - Airline (as a proxy for fleet efficiency)
       - Route characteristics
    
    4. **Airline Reliability Predictor**
    
       This model would predict the likelihood of delays or cancellations by airline and route. Key features:
       - Historical delay data
       - Weather considerations
       - Airport congestion
       - Time of day/year
    """)
    
    # Implementation plan
    st.markdown("""
    ### 🔧 Model Implementation Plan
    
    To build the advanced models mentioned above, I recommend the following approach:
    
    1. **Data Collection**:
       - Expand your dataset with historical pricing data for the same routes
       - Add weather data for common departure/arrival airports
       - Include airline on-time performance statistics
       - Add more detailed aircraft information
    
    2. **Feature Engineering**:
       - Create time-based features (day of year, holiday indicators)
       - Calculate price volatility metrics
       - Engineer route popularity features
       - Create airline quality scores from multiple factors
    
    3. **Model Development**:
       - For price prediction: Gradient Boosting Regressor (already implemented)
       - For price volatility: Time series models (ARIMA, Prophet)
       - For emissions: Regression models with aircraft-specific features
       - For reliability: Classification models for delay probability
    
    4. **Integration**:
       - Combine model outputs into a single recommendation system
       - Create an automated alert system for optimal booking times
       - Develop visualization tools for understanding trade-offs
    
    5. **Deployment**:
       - Deploy models as REST APIs
       - Create a dashboard for monitoring predictions
       - Implement feedback mechanisms to improve models over time
    """)
    
    # Code examples
    with st.expander("💻 Sample Code for Advanced Models"):
        st.code('''
# Price Volatility Model (Time Series)
import pandas as pd
from prophet import Prophet

def build_price_volatility_model(historical_price_data):
    # Prepare data in Prophet format
    df = historical_price_data.rename(columns={'date': 'ds', 'price': 'y'})
    
    # Create and fit model
    model = Prophet(
        changepoint_prior_scale=0.05,  # More flexible trend changes
        seasonality_prior_scale=10,    # Stronger seasonality
        daily_seasonality=True         # Daily price fluctuations
    )
    
    # Add weekly seasonality explicitly
    model.add_seasonality(name='weekly', period=7, fourier_order=3)
    
    # Fit model
    model.fit(df)
    
    # Create future dataframe for predictions
    future = model.make_future_dataframe(periods=30)  # Forecast 30 days
    
    # Make predictions
    forecast = model.predict(future)
    
    return model, forecast

# Carbon Emissions Model (using flight characteristics)
from sklearn.ensemble import RandomForestRegressor

def build_emissions_model(flight_data):
    # Prepare features
    X = flight_data[['distance_km', 'passengers', 'is_wide_body', 'avg_fleet_age']]
    X = pd.get_dummies(X, columns=['aircraft_type'])
    
    # Target: carbon emissions
    y = flight_data['carbon_emissions']
    
    # Create and train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

# Airline Reliability Model
from sklearn.ensemble import GradientBoostingClassifier

def build_reliability_model(flight_data):
    # Features affecting reliability
    X = flight_data[[
        'airline', 'departure_airport', 'arrival_airport',
        'time_of_day', 'day_of_week', 'month', 
        'is_holiday', 'avg_airport_congestion'
    ]]
    X = pd.get_dummies(X, columns=['airline', 'departure_airport', 'arrival_airport'])
    
    # Target: flight delayed (1) or on time (0)
    y = flight_data['is_delayed']
    
    # Create and train model
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

# Combined Recommendation Engine
def get_flight_recommendations(
    flights_df, 
    price_model, 
    volatility_model, 
    emissions_model, 
    reliability_model,
    price_weight=0.4,
    volatility_weight=0.2,
    emissions_weight=0.2,
    reliability_weight=0.2
):
    # Get predictions from each model
    flights_df['predicted_price'] = price_model.predict(price_features)
    flights_df['price_volatility'] = volatility_model.predict(volatility_features)['trend_increasing']
    flights_df['predicted_emissions'] = emissions_model.predict(emissions_features)
    flights_df['reliability_score'] = 1 - reliability_model.predict_proba(reliability_features)[:, 1]
    
    # Scale all factors to 0-1 range
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    flights_df[['price_scaled', 'volatility_scaled', 'emissions_scaled', 'reliability_scaled']] = scaler.fit_transform(
        flights_df[['predicted_price', 'price_volatility', 'predicted_emissions', 'reliability_score']]
    )
    
    # Calculate weighted score (lower is better for price, volatility, emissions)
    flights_df['weighted_score'] = (
        flights_df['price_scaled'] * price_weight +
        flights_df['volatility_scaled'] * volatility_weight +
        flights_df['emissions_scaled'] * emissions_weight -  # Negative because higher reliability is better
        flights_df['reliability_scaled'] * reliability_weight
    )
    
    # Return top recommended flights
    return flights_df.sort_values('weighted_score')
''')

    # Real-time monitoring section
    st.header("📊 Real-time Price Monitoring System")
    
    st.markdown("""
    To maximize the value of these predictive models, you should implement a real-time price monitoring system that:
    
    1. **Tracks price changes**: Monitors prices for routes you're interested in
    2. **Sets alerts**: Notifies you when prices hit predefined thresholds
    3. **Predicts optimal booking windows**: Recommends the best time to book
    
    ### Example Implementation
    
    ```python
    def monitor_flight_prices(route, departure_date, price_threshold):
        \"\"\"
        Monitor flight prices and alert when conditions are favorable
        \"\"\"
        # Set up monitoring parameters
        monitoring_active = True
        check_frequency = 6  # hours
        
        # Configure prediction model
        price_model = load_trained_price_model()
        volatility_model = load_trained_volatility_model()
        
        while monitoring_active:
            # Fetch current prices
            current_prices = fetch_current_prices(route, departure_date)
            
            # Predict price trend
            days_to_departure = (departure_date - datetime.now().date()).days
            price_prediction = predict_price_trend(route, days_to_departure)
            
            # Check conditions and send alerts
            if current_prices['min_price'] <= price_threshold:
                send_alert(f"Price alert: {route} is now {current_prices['min_price']}")
                monitoring_active = False
            
            elif price_prediction['trend'] == 'decreasing' and price_prediction['confidence'] > 0.7:
                send_alert(f"Hold recommendation: {route} prices likely to decrease soon")
            
            elif price_prediction['trend'] == 'increasing' and price_prediction['confidence'] > 0.7:
                send_alert(f"Book now recommendation: {route} prices expected to increase")
            
            # Wait for next check
            time.sleep(check_frequency * 3600)
    ```
    """)
    
    # Final actionable insights
    st.header("🚀 Actionable Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Optimization")
        st.markdown("""
        - Book flights on **Wednesdays**, **Thursdays**, or **Fridays** for the lowest prices
        - The best time to book is either **0-7 days** or **30+ days** before departure
        - **Night** and **late evening** flights tend to be cheaper
        - **Condor**, **Air Europa**, and **Norse Atlantic** typically offer the lowest average prices
        """)
    
    with col2:
        st.subheader("Emissions Reduction")
        st.markdown("""
        - Choose airlines with newer fleets for lower carbon emissions
        - Non-stop flights generally have lower total emissions than multi-stop routes
        - Airlines with the lowest emissions include **easyJet**, **Norwegian**, and **JetBlue**
        - When comparing similar flights, check the aircraft type - newer models are more fuel-efficient
        """)
    
    st.markdown("""
    ### Custom Recommendation
    
    Based on your preference settings (Price: {:.0%}, Emissions: {:.0%}), your ideal booking strategy is:
    """.format(price_weight, emission_weight))
    
    if price_weight > 0.7:
        st.info("Focus primarily on **price optimizations** while making reasonable compromises on emissions. Look for deals from budget carriers and be flexible with your travel dates and times.")
    elif emission_weight > 0.7:
        st.info("Prioritize **emissions reductions** while accepting potentially higher prices. Choose airlines with newer, more efficient fleets and direct routes even if they cost somewhat more.")
    else:
        st.info("Take a **balanced approach** by selecting airlines that score well on both metrics. Consider paying slightly more for significantly lower emissions, but avoid the highest-priced options.")
    
else:
    st.error("No data available after applying filters. Please adjust your filter criteria.")
