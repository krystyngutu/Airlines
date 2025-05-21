import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import datetime
import calendar

# ----------------------
# CONFIGURATION
# ----------------------
DATA_PATH = r"C:/Users/kryst/OneDrive/Desktop/Swiss Airlines/airlines/all_flights.csv"

DIRECT_AIRLINES = ['SWISS', 'United', 'Delta']
LUFTHANSA_GROUP = [
    'Austrian', 'Brussels Airlines', 'Discover Airlines', 'Eurowings',
    'Edelweiss Air', 'ITA', 'Air Dolomiti', 'Lufthansa', 'SWISS'
]
STAR_ALLIANCE = [
    'Aegean', 'Air Canada', 'Air China', 'Air India', 'Air New Zealand',
    'ANA', 'Asiana Airlines', 'Austrian', 'Avianca', 'Brussels Airlines',
    'CopaAirlines', 'Croatia Airlines', 'Egyptair', 'Ethiopian Airlines',
    'Eva Air', 'LOT Polish Airlines', 'Lufthansa', 'Shenzhen Airlines',
    'Singapore Airlines', 'South African Airways', 'SWISS', 'Tap Air Portugal',
    'Thai', 'Turkish Airlines', 'United'
]

AIRLINE_COLORS = {
    'Lufthansa': '#ffd700', 'SWISS': '#d71920', 'Delta': '#00235f', 'United': '#1a75ff',
    'Edelweiss Air': '#800080', 'Air Dolomiti': '#32cd32', 'Austrian': '#c3f550',
    'ITA': '#fbaa3f', 'Brussels Airlines': '#00235f', 'Eurowings': '#1a75ff',
    'Aegean': '#767676', 'Air Canada': '#00235f', 'Tap Air Portugal': '#fbaa3f',
    'Turkish Airlines': '#800080'
}

# ----------------------
# DATA LOADING & FEATURE ENGINEERING
# ----------------------
@st.cache_data
def load_and_prepare(path=DATA_PATH):
    df = pd.read_csv(path)
    # parse and clean
    df['departureTime'] = pd.to_datetime(df['departureTime'], errors='coerce')
    df['price'] = np.ceil(pd.to_numeric(df['price'], errors='coerce'))
    df['duration'] = pd.to_numeric(df.get('durationTime', np.nan), errors='coerce')
    df['carbon'] = pd.to_numeric(df.get('carbonEmissionsThisFlight', np.nan), errors='coerce')
    df['airline'] = df.get('airline', '').fillna('Unknown').str.strip()
    df = df.dropna(subset=['departureTime', 'price', 'airline'])

    # datetime features
    df['hour'] = df['departureTime'].dt.hour
    df['weekday'] = df['departureTime'].dt.day_name()
    df['dayOfWeek'] = df['departureTime'].dt.weekday
    df['month'] = df['departureTime'].dt.month
    df['season'] = pd.cut(
        df['month'],
        bins=[0,2,5,8,11,12],
        labels=['Winter','Spring','Summer','Fall','Winter'],
        right=True, include_lowest=True
    )
    df['travelClass'] = df.get('travelClass','Unknown').fillna('Unknown').str.strip()
    df['numLayovers'] = df.get('layovers','').fillna('').apply(lambda s: 0 if s=='' else s.count(',')+1)
    df['timeOfDay'] = pd.cut(
        df['hour'],
        bins=[-1,4,11,16,21,24],
        labels=['Night','Morning','Afternoon','Evening','Night'],
        right=True
    )
    return df

# ----------------------
# STREAMLIT SETUP
# ----------------------
st.set_page_config(layout="wide")
st.title("Aviation Revenue Steering Analysis: NYC → CH (May 2025–Mar 2026)")

df = load_and_prepare()
# Sidebar filters
st.sidebar.header("Filters")
price_min, price_max = int(df['price'].min()), int(df['price'].max())
price_range = st.sidebar.slider("Price Range (USD)", price_min, price_max, (price_min, price_max))
group = st.sidebar.radio("Airline Group", ["All","Direct","Lufthansa","Star"])

mask = df['price'].between(*price_range)
if group == "Direct": mask &= df['airline'].isin(DIRECT_AIRLINES)
elif group == "Lufthansa": mask &= df['airline'].isin(LUFTHANSA_GROUP)
elif group == "Star": mask &= df['airline'].isin(STAR_ALLIANCE)
df = df[mask]

# ----------------------
# PLOTTING HELPERS
# ----------------------
def plot_bar(df, x, y, title, labels, category_order=None, color=None, text_auto=True):
    fig = px.bar(df, x=x, y=y, title=title, labels=labels,
                 text_auto=text_auto, color=color,
                 color_discrete_map=AIRLINE_COLORS if color=='airline' else None)
    if category_order:
        fig.update_layout(xaxis={'categoryorder':'array','categoryarray':category_order})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# Price Analysis
st.header("Price Analysis")
# Weekly trend
weekly = df.groupby([pd.Grouper(key='departureTime', freq='W'), 'airline'])['price'].mean().reset_index()
weekly['price'] = np.ceil(weekly['price'])
fig = px.line(weekly, x='departureTime', y='price', color='airline',
              labels={'departureTime':'Week','price':'Price (USD)'},
              color_discrete_map=AIRLINE_COLORS)
st.plotly_chart(fig, use_container_width=True)

# Monthly & Seasonal
col1, col2 = st.columns(2)
with col1:
    mon = df.groupby('month')['price'].mean().reindex(range(1,13)).reset_index()
    mon['price'] = np.ceil(mon['price'])
    mon['month_name'] = mon['month'].apply(lambda m: calendar.month_abbr[m])
    plot_bar(mon, 'month_name','price','Avg Price by Month',{'month_name':'Month','price':'Price (USD)'})
with col2:
    sea = df.groupby('season')['price'].mean().reindex(['Winter','Spring','Summer','Fall']).reset_index()
    sea['price']=np.ceil(sea['price'])
    plot_bar(sea,'season','price','Avg Price by Season',{'season':'Season','price':'Price (USD)'})

# Day & Time
col3, col4 = st.columns(2)
with col3:
    dow = df.groupby('weekday')['price'].mean().reindex(
        ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    ).reset_index()
    dow['price']=np.ceil(dow['price'])
    plot_bar(dow,'weekday','price','Avg Price by Day',{'weekday':'Day','price':'Price (USD)'})
with col4:
    tod = df.groupby('timeOfDay')['price'].mean().reindex(['Morning','Afternoon','Evening','Night']).reset_index()
    tod['price']=np.ceil(tod['price'])
    plot_bar(tod,'timeOfDay','price','Avg Price by Time of Day',{'timeOfDay':'Time','price':'Price (USD)'},
             category_order=['Morning','Afternoon','Evening','Night'])

# Layovers & Class
col5, col6 = st.columns(2)
with col5:
    lay = df.groupby('numLayovers')['price'].mean().reset_index(); lay['price']=np.ceil(lay['price'])
    plot_bar(lay,'numLayovers','price','Avg Price by Layovers',{'numLayovers':'Layovers','price':'Price (USD)'})
with col6:
    tc = df.groupby('travelClass')['price'].mean().reset_index(); tc['price']=np.ceil(tc['price'])
    plot_bar(tc,'travelClass','price','Avg Price by Travel Class',{'travelClass':'Class','price':'Price (USD)'})

# Airline comparison
air = df.groupby('airline')['price'].mean().reset_index(); air['price']=np.ceil(air['price'])
plot_bar(air,'airline','price','Avg Price by Airline',{'airline':'Airline','price':'Price (USD)'},color='airline')

# ----------------------
# MODEL UTILS & TRAINING
# ----------------------
@st.cache_data
def get_features(df):
    df = df.dropna(subset=['dayOfWeek','hour','month','duration','carbon','price'])
    X = df[['dayOfWeek','hour','month','duration','carbon','airline']].copy()
    y = df['price']
    return X, y

# Define preprocessor
num_feats = ['dayOfWeek','hour','month','duration','carbon']
cat_feats = ['airline']
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_feats),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
])

def train_evaluate(model, X_train, X_test, y_train, y_test, name):
    pipeline = Pipeline([('prep', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return pipeline, rmse, r2

# Prepare data
X, y = get_features(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model definitions with auto-CV where possible
models = {
    'Linear': LinearRegression(),
    'RidgeCV': RidgeCV(alphas=[0.1,1.0,10.0], cv=5),
    'LassoCV': LassoCV(alphas=[0.01,0.1,1.0], cv=5),
    'ElasticNetCV': ElasticNetCV(alphas=[0.01,0.1,1.0], l1_ratio=[.2,.5,.8], cv=5),
    'RandForest': RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        {'n_estimators':[100,200],'max_depth':[None,10,20]},
        n_iter=4,cv=3,random_state=42
    ),
    'HistGB': HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, early_stopping=True)
}

results = {}
best_name, best_rmse = None, float('inf')
for name, model in models.items():
    pipe, rmse, r2 = train_evaluate(model, X_train, X_test, y_train, y_test, name)
    results[name] = rmse
    st.metric(f"{name} RMSE", f"USD{rmse:.2f}", delta=None)
    if rmse < best_rmse:
        best_rmse, best_name, best_pipe = rmse, name, pipe

# Summary plot
st.subheader("Model RMSE Comparison")
fig = px.bar(x=list(results.keys()), y=list(results.values()),
             labels={'x':'Model','y':'RMSE'}, text_auto=True)
st.plotly_chart(fig, use_container_width=True)
st.success(f"Best model: {best_name} (RMSE USD{best_rmse:.2f})")

# ----------------------
# PREDICTIONS HEATMAP
# ----------------------
st.header("Revenue Optimization Insights")
current_month = datetime.datetime.now().month
common_air = df['airline'].mode()[0]
avg_dur = df['duration'].mean()

preds = []
for day in range(7):
    for hr in range(24):
        sample = pd.DataFrame({
            'dayOfWeek':[day],'hour':[hr],'month':[current_month],
            'duration':[avg_dur],'carbon':[df['carbon'].mean()],
            'airline':[common_air]
        })
        price = best_pipe.predict(sample)[0]
        preds.append({'day':day,'hour':hr,'price':price})
pd_df = pd.DataFrame(preds)
pd_df['day_name'] = pd_df['day'].map(
    dict(enumerate(['Mon','Tue','Wed','Thu','Fri','Sat','Sun']))
)
heat = pd_df.pivot('day_name','hour','price')
fig = px.imshow(heat, labels={'x':'Hour','y':'Day','color':'Price (USD)'})
st.plotly_chart(fig, use_container_width=True)
