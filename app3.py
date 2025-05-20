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
st.title("Flight Price Exploration (NYC to CH): Revenue Steering Analysis")

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
    df['dayOfWeek'] = df['departureTime'].dt.weekday
    df['hour'] = df['departureTime'].dt.hour
    df['month'] = df['departureTime'].dt.month
    df['airline'] = df['airline'].astype(str).str.strip()
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

# Load and filter data
df = load_data()

# ROUTE FILTERING: NYC to SWITZERLAND
nyc_airports = ["LGA", "JFK", "EWR"]
swiss_airports = ["ZRH", "BSL", "GVA"]

if 'departureAirportID' in df.columns and 'arrivalAirportID' in df.columns:
    df = df[df['departureAirportID'].isin(nyc_airports) & df['arrivalAirportID'].isin(swiss_airports)]

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
st.sidebar.header("Airline Filters")

# Define airline groups
direct_airlines = ['SWISS', 'United', 'Delta']
lufthansa_group = ['Austrian', 'Brussels Airlines', 'Discover Airlines', 'Eurowings', 'Edelweiss Air', 'ITA', 'Air Dolomiti', 'Lufthansa', 'SWISS']
star_alliance = ['Aegean', 'Air Canada', 'Air China', 'Air India', 'Air New Zealand', 'ANA', 'Asiana Airlines',
    'Austrian', 'Avianca', 'Brussels Airlines', 'CopaAirlines', 'Croatia Airlines', 'Egyptair',
    'Ethiopian Airlines', 'Eva Air', 'LOT Polish Airlines', 'Lufthansa', 'Shenzhen Airlines',
    'Singapore Airlines', 'South African Airways', 'SWISS', 'Tap Air Portugal', 'Thai',
    'Turkish Airlines', 'United']

# Radio button filter
airline_group = st.sidebar.radio(
    "Select Flight Group:",
    options=["All Flights", "Direct Airlines", "Lufthansa Group", "Star Alliance"]
)

# Apply airline group filter
if airline_group == "Direct Airlines":
    df = df[df["airline"].isin(direct_airlines)]
elif airline_group == "Lufthansa Group":
    df = df[df["airline"].isin(lufthansa_group)]
elif airline_group == "Star Alliance":
    df = df[df["airline"].isin(star_alliance)]

# Price slider
min_price = int(df['price'].min())
max_price = int(df['price'].max())

price_range = st.sidebar.slider(
    "Price Range ($)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price)
)

# Apply price filter
df = df[
    (df['price'] >= price_range[0]) & (df['price'] <= price_range[1])
]
min_price = int(df['price'].min())
max_price = int(df['price'].max())

# Apply price filter
df = df[
    (df['price'] >= price_range[0]) & (df['price'] <= price_range[1])
]

# ----------------------
# PRICE ANALYSIS
# ----------------------
st.header("Price Analysis")
col1, col2 = st.columns(2)

with col1:
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_day = df.groupby('weekday')['price'].mean().reindex(day_order).reset_index()
    fig = px.bar(df_day, x='weekday', y='price', title='Average Price by Day of Week',
                 labels={'price': 'Avg Price ($)', 'weekday': 'Day'}, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"💰 Cheapest day to fly: **{df_day.loc[df_day['price'].idxmin(), 'weekday']}**")

with col2:
    tod_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    df_tod = df.groupby('timeOfDay')['price'].mean().reindex(tod_order).reset_index()
    fig = px.bar(df_tod, x='timeOfDay', y='price', title='Average Price by Time of Day',
             labels={'price': 'Avg Price ($)', 'timeOfDay': 'Time'}, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"💰 Cheapest time to fly: **{df_tod.loc[df_tod['price'].idxmin(), 'timeOfDay']}**")
    st.caption("🕐 Morning: 5am–12pm, Afternoon: 12–5pm, Evening: 5–10pm, Night: 10pm–5am")

st.subheader("Airline Price Comparison")
df_airline = df.groupby('airline')['price'].mean().reset_index()
fig = px.bar(df_airline, x='airline', y='price', color='airline',
             color_discrete_map=airline_colors,
             title='Average Price by Airline',
             labels={'price': 'Average Price ($)'}, text_auto=True)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

st.header("Revenue Steering Models")
st.markdown("""
Revenue management and pricing teams use these models to optimize flight pricing strategy:
- **Linear models**: Baseline for understanding price drivers
- **Regularized models (Ridge, Lasso, ElasticNet)**: Control for overfitting in dynamic pricing
- **Ensemble models (Random Forest, Gradient Boosting)**: Capture complex patterns for demand forecasting
""")

# Prepare modeling data
@st.cache_data
def prepare_model_data(df):
    df['wifiEncoded'] = df['wifi'].fillna('Unknown').astype('category').cat.codes
    df['airplaneEncoded'] = df['airplane'].fillna('Unknown').astype('category').cat.codes
    features = ['dayOfWeek', 'hour', 'month', 'airline', 'durationTime', 'carbonEmissionsThisFlight', 'wifiEncoded', 'airplaneEncoded']
    target = 'price'
    
    # Convert categorical features to numeric
    X = df[features].copy()
    y = df[target]
    
    return X, y

# Create models tab system
model_tab1, model_tab2, model_tab3 = st.tabs(["Linear Models", "Regularized Models", "Ensemble Models"])

try:
    X, y = prepare_model_data(df)
    
    # Handle categorical variables
    categorical_features = ['airline']
    numerical_features = ['dayOfWeek', 'hour', 'month', 'durationTime']
    
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
        st.text("The linear model is underperforming due to the data having nonlinear patterns and interactions.")
    
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
        Regularized linear regression models are designed to prevent overfitting (when a model learns the data too well) by penalizing large coefficients.
        """)
    
        st.markdown("""
            **Revenue Management Applications:**
            - **Ridge**: Controls for multicollinearity between features (common in seasonal data); adds the squared magnitude of the coefficients to the loss function
            - **Lasso**: Feature selection for dynamic pricing models; adds the absolute value of coefficients to the loss function
            - **ElasticNet**: Hybrid approach for balanced feature selection and coefficient shrinkage; combines Ridge and Lasso
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
    st.header("Revenue Optimization Insights")
    
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
                'dayOfWeek': [day],
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


# ----------------------
# ADDITIONAL ANALYTICS
# ----------------------
st.header("Operational Feature Analysis")

col3, col4 = st.columns(2)

with col3:
    if 'carbonEmissionsThisFlight' in df.columns:
        df_carbon = df.dropna(subset=['carbonEmissionsThisFlight'])
        fig = px.box(df_carbon, x='airline', y='carbonEmissionsThisFlight', color='airline',
                     color_discrete_map=airline_colors,
                     title='Carbon Emissions by Airline',
                     labels={'carbonEmissionsThisFlight': 'Carbon Emissions (kg)'})
        st.plotly_chart(fig, use_container_width=True)
st.markdown('---')
if 'airplane' in df.columns:
        df_aircraft = df.dropna(subset=['airplane'])
        fig = px.box(df_aircraft, x='airplane', y='price', title='Price by Aircraft Type',
                     labels={'price': 'Price ($)', 'airplane': 'Aircraft'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
with col4:
    if 'legroom' in df.columns:
        df_legroom = df.dropna(subset=['legroom'])
        df_legroom['legroom'] = pd.to_numeric(df_legroom['legroom'].str.extract(r'(\d+)')[0], errors='coerce')
        fig = px.box(df_legroom, x='airline', y='legroom', color='airline',
                     color_discrete_map=airline_colors,
                     title='Legroom by Airline',
                     labels={'legroom': 'Legroom (in)'})
        st.plotly_chart(fig, use_container_width=True)

    
# ----------------------
# ADVANCED MODELING WITH OPERATIONAL FEATURES
# ----------------------
st.header("Advanced Modeling with Operational Features")

# Feature engineering
df['wifiEncoded'] = df['wifi'].fillna('Unknown').astype('category').cat.codes
df['airplaneEncoded'] = df['airplane'].fillna('Unknown').astype('category').cat.codes
if 'legroom' in df.columns:
    df['legroom'] = pd.to_numeric(df['legroom'].str.extract(r'(\d+)')[0], errors='coerce')
else:
    df['legroom'] = np.nan

advanced_features = ['dayOfWeek', 'hour', 'month', 'durationTime', 'carbonEmissionsThisFlight',
                     'wifiEncoded', 'airplaneEncoded', 'legroom']
categorical_features = []
numerical_features = ['dayOfWeek', 'hour', 'month', 'durationTime', 'carbonEmissionsThisFlight', 'wifiEncoded', 'airplaneEncoded', 'legroom']

# Drop rows with missing advanced features
df_model_ready = df.dropna(subset=numerical_features + ['price'])

X_adv = df_model_ready[advanced_features]
y_adv = df_model_ready['price']

# Train/test split
X_train_adv, X_test_adv, y_train_adv, y_test_adv = train_test_split(X_adv, y_adv, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_adv)
X_test_scaled = scaler.transform(X_test_adv)

# Train additional models
model_results = {}

for name, model in {
    'Linear (Adv)': LinearRegression(),
    'Ridge (Adv)': Ridge(alpha=1.0),
    'Random Forest (Adv)': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting (Adv)': GradientBoostingRegressor(n_estimators=100, random_state=42)
}.items():
    model.fit(X_train_scaled, y_train_adv)
    preds = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test_adv, preds))
    r2 = r2_score(y_test_adv, preds)
    model_results[name] = rmse
    st.metric(f"{name} RMSE", f"${rmse:.2f}")
    st.caption(f"{name} R²: {r2:.4f}")

# Summary plot
fig = px.bar(
    x=list(model_results.keys()),
    y=list(model_results.values()),
    title="Advanced Model RMSE Comparison",
    labels={'x': 'Model', 'y': 'RMSE'},
    color_discrete_sequence=['#1a75ff'] * len(model_results)
)
fig.update_traces(texttemplate='$%{y:.2f}', textposition='outside')
st.plotly_chart(fig, use_container_width=True)

# ----------------------
# FEATURE IMPORTANCE VISUALIZATION
# ----------------------
st.header("Feature Importance from Advanced Models")

# Use Random Forest for importance (or Gradient Boosting if preferred)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train_adv)

feature_names = advanced_features
importances = rf_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

fig = px.bar(
    importance_df,
    x='Importance',
    y='Feature',
    orientation='h',
    title='Feature Importance (Random Forest)',
    labels={'Importance': 'Relative Importance', 'Feature': 'Feature'},
    color_discrete_sequence=['#1a75ff']
)
st.plotly_chart(fig, use_container_width=True)
