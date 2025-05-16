import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import datetime

# ----------------------
# PAGE SETUP
# ----------------------
st.set_page_config(layout="wide")
st.title("🛫 Flight Price Explorer: Revenue Steering Analysis")

# ----------------------
# LOAD & CLEAN DATA
# ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("all_flights.csv")
    df['departureTime'] = pd.to_datetime(df['departureTime'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['durationTime'] = pd.to_numeric(df['durationTime'], errors='coerce')
    df['weekday'] = df['departureTime'].dt.day_name()
    df['day_of_week'] = df['departureTime'].dt.weekday
    df['hour'] = df['departureTime'].dt.hour
    df['month'] = df['departureTime'].dt.month
    if 'wifi' not in df.columns:
        df['wifi'] = 'Unknown'

    def time_of_day(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'
    df['timeOfDay'] = df['hour'].apply(time_of_day)
    return df.dropna(subset=['price', 'airline'])

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ----------------------
# COLOR PALETTE
# ----------------------
airline_colors = {
    'Lufthansa': '#ffd700',
    'SWISS': '#d71920',
    'Delta': '#00235f',
    'United': '#1a75ff',
    'Edelweiss Air': '#800080',
    'Air Dolomiti': '#32cd32',
    'Austrian': '#c3f550',
    'ITA': '#fbaa3f',
    'Brussels Airlines': '#00235f',
    'Eurowings': '#1a75ff',
    'Aegean': '#767676',
    'Air Canada': '#00235f',
    'Tap Air Portugal': '#fbaa3f',
    'Turkish Airlines': '#800080'
}
custom_colors = ['#d71920', '#00235f', '#f9ba00', '#660000', '#800080', '#3366ff',
                 '#c3f550', '#fbaa3f', '#000000']

# ----------------------
# SIDEBAR FILTERS
# ----------------------
st.sidebar.header("Filters")
direct_airlines = ['SWISS', 'United', 'Delta']
lufthansa_group = ['Austrian', 'Brussels Airlines', 'Discover Airlines', 'Eurowings',
                   'Edelweiss Air', 'ITA', 'Air Dolomiti', 'Lufthansa', 'SWISS']
star_alliance = ['Aegean', 'Air Canada', 'Air China', 'Air India', 'Air New Zealand', 'ANA',
                 'Asiana Airlines', 'Austrian', 'Avianca', 'Brussels Airlines', 'CopaAirlines',
                 'Croatia Airlines', 'Egyptair', 'Ethiopian Airlines', 'Eva Air', 'LOT Polish Airlines',
                 'Lufthansa', 'Shenzhen Airlines', 'Singapore Airlines', 'South African Airways',
                 'SWISS', 'Tap Air Portugal', 'Thai', 'Turkish Airlines', 'United']
group_option = st.sidebar.radio("Airline Group", ['All Airlines', 'Direct Airlines', 'Lufthansa Group', 'Star Alliance'])

if group_option == 'Direct Airlines':
    airline_filter = direct_airlines
elif group_option == 'Lufthansa Group':
    airline_filter = lufthansa_group
elif group_option == 'Star Alliance':
    airline_filter = star_alliance
else:
    airline_filter = sorted(df['airline'].unique())

df_filtered = df[df['airline'].isin(airline_filter)]

min_price = int(df_filtered['price'].min())
max_price = int(df_filtered['price'].max())
price_range = st.sidebar.slider("Price Range ($)", min_value=min_price, max_value=max_price, value=(min_price, max_price))
df_filtered = df_filtered[(df_filtered['price'] >= price_range[0]) & (df_filtered['price'] <= price_range[1])]

# ----------------------
# PRICE ANALYSIS
# ----------------------
st.header("📊 Price Analysis")
col1, col2 = st.columns(2)

with col1:
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_day = df_filtered.groupby('weekday')['price'].mean().reindex(day_order).reset_index()
    fig = px.bar(df_day, x='weekday', y='price', title='Average Price by Day of Week',
                 labels={'price': 'Avg Price ($)', 'weekday': 'Day'}, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"💰 Cheapest day to fly: **{df_day.loc[df_day['price'].idxmin(), 'weekday']}**")

with col2:
    tod_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    df_tod = df_filtered.groupby('timeOfDay')['price'].mean().reindex(tod_order).reset_index()
    fig = px.bar(df_tod, x='timeOfDay', y='price', title='Average Price by Time of Day',
                 labels={'price': 'Avg Price ($)', 'timeOfDay': 'Time'}, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"💰 Cheapest time to fly: **{df_tod.loc[df_tod['price'].idxmin(), 'timeOfDay']}**")

st.subheader("Airline Price Comparison")
df_airline = df_filtered.groupby('airline')['price'].mean().reset_index()
fig = px.bar(df_airline, x='airline', y='price', color='airline',
             color_discrete_map=airline_colors,
             title='Average Price by Airline',
             labels={'price': 'Average Price ($)'}, text_auto=True)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

st.header("📈 Revenue Steering Models")
st.markdown("""
Revenue management and pricing teams use these models to optimize flight pricing strategy:
- **Linear models**: Baseline for understanding price drivers
- **Regularized models (Ridge, Lasso, ElasticNet)**: Control for overfitting in dynamic pricing
- **Ensemble models (Random Forest, Gradient Boosting)**: Capture complex patterns for demand forecasting
""")

# Prepare modeling data
@st.cache_data
def prepare_model_data(df):
    df['wifi_encoded'] = df['wifi'].fillna('Unknown').astype('category').cat.codes
    df['airplane_encoded'] = df['airplane'].fillna('Unknown').astype('category').cat.codes
    features = ['day_of_week', 'hour', 'month', 'airline', 'durationTime', 'carbonEmissionsThisFlight', 'wifi_encoded', 'airplane_encoded']
    target = 'price'
    
    # Convert categorical features to numeric
    X = df[features].copy()
    y = df[target]
    
    return X, y

# Create models tab system
model_tab1, model_tab2, model_tab3 = st.tabs(["Linear Models", "Regularized Models", "Ensemble Models"])

try:
    X, y = prepare_model_data(df_filtered)
    
    # Handle categorical variables
    categorical_features = ['airline']
    numerical_features = ['day_of_week', 'hour', 'month', 'durationTime']
    
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with model_tab1:
        st.subheader("Linear Regression Models")
        
        # Linear Regression
        lr_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])
        
        lr_pipeline.fit(X_train, y_train)
        lr_preds = lr_pipeline.predict(X_test)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
        lr_r2 = r2_score(y_test, lr_preds)
        
        st.metric("Linear Regression RMSE", f"${lr_rmse:.2f}")
        st.metric("Linear Regression R²", f"{lr_r2:.4f}")
        
        # Feature importance for linear model (using coefficients)
        # This is simplified and would need more processing for actual feature importance
        st.text("Linear model helps understand the baseline price drivers")
    
    with model_tab2:
        st.subheader("Regularized Models")
        
        col1, col2, col3 = st.columns(3)
        
        # Ridge Regression
        with col1:
            ridge_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', Ridge(alpha=1.0))
            ])
            
            ridge_pipeline.fit(X_train, y_train)
            ridge_preds = ridge_pipeline.predict(X_test)
            ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_preds))
            ridge_r2 = r2_score(y_test, ridge_preds)
            
            st.metric("Ridge RMSE", f"${ridge_rmse:.2f}")
            st.metric("Ridge R²", f"{ridge_r2:.4f}")
        
        # Lasso Regression
        with col2:
            lasso_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', Lasso(alpha=0.1))
            ])
            
            lasso_pipeline.fit(X_train, y_train)
            lasso_preds = lasso_pipeline.predict(X_test)
            lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_preds))
            lasso_r2 = r2_score(y_test, lasso_preds)
            
            st.metric("Lasso RMSE", f"${lasso_rmse:.2f}")
            st.metric("Lasso R²", f"{lasso_r2:.4f}")
        
        # ElasticNet
        with col3:
            en_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', ElasticNet(alpha=0.1, l1_ratio=0.5))
            ])
            
            en_pipeline.fit(X_train, y_train)
            en_preds = en_pipeline.predict(X_test)
            en_rmse = np.sqrt(mean_squared_error(y_test, en_preds))
            en_r2 = r2_score(y_test, en_preds)
            
            st.metric("ElasticNet RMSE", f"${en_rmse:.2f}")
            st.metric("ElasticNet R²", f"{en_r2:.4f}")
        
        st.markdown("""
        **Revenue Management Applications:**
        - **Ridge**: Controls for multicollinearity between features (common in seasonal data)
        - **Lasso**: Feature selection for dynamic pricing models
        - **ElasticNet**: Hybrid approach for balanced feature selection and coefficient shrinkage
        """)
    
    with model_tab3:
        st.subheader("Ensemble Models")
        
        col1, col2 = st.columns(2)
        
        # Random Forest
        with col1:
            rf_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            rf_pipeline.fit(X_train, y_train)
            rf_preds = rf_pipeline.predict(X_test)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
            rf_r2 = r2_score(y_test, rf_preds)
            
            st.metric("Random Forest RMSE", f"${rf_rmse:.2f}")
            st.metric("Random Forest R²", f"{rf_r2:.4f}")
            
            st.markdown("""
            **Revenue Management Application:**
            - Demand forecasting for different customer segments
            - Handles non-linear relationships in seasonal pricing
            """)
        
        # Gradient Boosting
        with col2:
            gb_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
            ])
            
            gb_pipeline.fit(X_train, y_train)
            gb_preds = gb_pipeline.predict(X_test)
            gb_rmse = np.sqrt(mean_squared_error(y_test, gb_preds))
            gb_r2 = r2_score(y_test, gb_preds)
            
            st.metric("Gradient Boosting RMSE", f"${gb_rmse:.2f}")
            st.metric("Gradient Boosting R²", f"{gb_r2:.4f}")
            
            st.markdown("""
            **Revenue Management Application:**
            - High-precision pricing optimization for premium routes
            - Real-time fare adjustment based on booking patterns
            """)
        
        # Compare model performance
        models = {
            'Linear Regression': lr_rmse,
            'Ridge': ridge_rmse,
            'Lasso': lasso_rmse,
            'ElasticNet': en_rmse,
            'Random Forest': rf_rmse,
            'Gradient Boosting': gb_rmse
        }
        
        best_model = min(models, key=models.get)
        
        st.success(f"✅ Best performing model: **{best_model}** with RMSE ${models[best_model]:.2f}")
        
        # Model comparison chart
        fig = px.bar(
            x=list(models.keys()),
            y=list(models.values()),
            labels={'x': 'Model', 'y': 'RMSE (lower is better)'},
            title='Model Performance Comparison'
        )
        fig.update_traces(texttemplate='$%{y:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    # Optimal booking recommendations
    st.header("💡 Revenue Optimization Insights")
    
    # Use the best model to predict prices for different scenarios
    best_pipeline = gb_pipeline if best_model == 'Gradient Boosting' else rf_pipeline

    # Find optimal day and hour
    all_days = range(7)  # 0 = Monday, 6 = Sunday
    all_hours = range(24)
    
    current_month = datetime.datetime.now().month
    
    # Create a sample flight for prediction (using most common values from data)
    most_common_airline = df['airline'].mode()[0]
    avg_duration = df['durationTime'].mean()
    
    # Generate price predictions for all day and hour combinations
    predictions = []
    for day in all_days:
        for hour in all_hours:
            # Create a test instance
            test_data = pd.DataFrame({
                'day_of_week': [day],
                'hour': [hour],
                'month': [current_month],
                'airline': [most_common_airline],
                'durationTime': [avg_duration]
            })
            
            # Predict price
            pred_price = best_pipeline.predict(test_data)[0]
            predictions.append({'day': day, 'hour': hour, 'price': pred_price})
    
    # Convert to DataFrame
    prediction_df = pd.DataFrame(predictions)
    
    # Map day numbers to names
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    prediction_df['day_name'] = prediction_df['day'].apply(lambda x: day_names[x])
    
    # Create a heatmap of predicted prices
    pivot_df = prediction_df.pivot(index='day_name', columns='hour', values='price')
    
    # Reorder days to start with Monday
    pivot_df = pivot_df.reindex(day_names)
    
    # Create heatmap
    fig = px.imshow(
        pivot_df,
        labels=dict(x="Hour of Day", y="Day of Week", color="Predicted Price ($)"),
        title=f"Predicted Prices by Day and Hour (for {most_common_airline})",
        color_continuous_scale="RdBu_r"
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Find the optimal booking time
    min_idx = prediction_df['price'].idxmin()
    optimal_day = prediction_df.loc[min_idx, 'day_name']
    optimal_hour = prediction_df.loc[min_idx, 'hour']
    min_price = prediction_df.loc[min_idx, 'price']
    
    st.success(f"""
    ### Revenue Steering Recommendations
    
    🎯 **For travelers seeking lowest fares**: 
    Book on **{optimal_day}** at **{optimal_hour}:00** (predicted price: ${min_price:.2f})
    
    💼 **For revenue management**:
    - Dynamic pricing should adjust for {optimal_day} bookings (lowest demand period)
    - Highest yield potential on {day_names[prediction_df.groupby('day')['price'].mean().idxmax()]}
    - Consider time-of-day fare differentiation with {int(prediction_df.groupby('hour')['price'].mean().idxmax())}:00 premium pricing
    """)

except Exception as e:
    st.error(f"Error in model building: {e}")
