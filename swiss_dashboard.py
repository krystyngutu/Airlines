import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Load JSON files
def load_json_files(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as file:
                try:
                    price_data = json.load(file)
                    allFlights = []
                    if 'bestFlights' in price_data.keys():
                        allFlights.append(price_data['bestFlights'])
                    if 'otherFlights' in price_data.keys():
                        allFlights.append(price_data['otherFlights'])
                    # allFlights = price_data['bestFlights'] + price_data['otherFlights']
                    for flight in allFlights:
                        all_data.append(flight)
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")
    return pd.DataFrame(all_data)

# Execute basic EDA
def basic_eda(df):
    print("\n--- BASIC EDA ---\n")
    print("DataFrame Shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nData Types:\n", df.dtypes)
    print("\nSample Data:\n", df.head())
    print("\nSummary Statistics:\n", df.describe(include='all'))

# Streamlit Dashboard
def streamlit_dashboard(df):
    st.set_page_config(page_title="SWISS Revenue Dashboard", layout="wide")
    st.title("Swiss Airlines vs Competitors Dashboard")

    st.sidebar.title("Filters")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(df['date']).min())
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(df['date']).max())
    df = df[(pd.to_datetime(df['date']) >= start_date) & (pd.to_datetime(df['date']) <= end_date)]

    origin = st.sidebar.selectbox("Select Origin Airport", options=sorted(df['origin'].dropna().unique()))
    destination = st.sidebar.selectbox("Select Destination Airport", options=sorted(df['destination'].dropna().unique()))
    df = df[(df['origin'] == origin) & (df['destination'] == destination)]

    col1, col2, col3 = st.columns(3)
    col1.metric("SWISS Cheaper (%)", f"{df['swiss_cheaper'].mean() * 100:.2f}%")
    col2.metric("Avg SWISS Price ($)", f"{df['swiss_price'].mean():.2f}")
    col3.metric("Price Outliers", df[df['swiss_price'] - df['lowest_competitor_price'] > 500].shape[0])

    tab1, tab2, tab3 = st.tabs(["Flight Overview", "Competitor Pricing", "Outliers"])

    with tab1:
        st.subheader("Basic Flight Info")
        st.dataframe(df[['origin', 'destination', 'airline', 'flight_number', 'departure_time', 'arrival_time']])

        st.subheader("Average Duration per Route")
        avg_duration = df.groupby(['origin', 'destination'])['duration_minutes'].mean()
        st.bar_chart(avg_duration)

        st.subheader("Seat Capacity per Aircraft")
        df['total_seats'] = df['seats_economy'] + df['seats_business'] + df['seats_first']
        st.bar_chart(df.groupby('aircraft_type')['total_seats'].mean())

    with tab2:
        st.subheader("Competitor Pricing Overview")
        comp_prices = []
        for comp_list in df['competitor_prices']:
            if isinstance(comp_list, list):
                comp_prices.extend(comp_list)
        competitor_df = pd.DataFrame(comp_prices)

        if not competitor_df.empty:
            avg_prices = competitor_df.groupby('competing_airline')['price_usd'].mean().sort_values()
            st.bar_chart(avg_prices)

            st.subheader("Top 5 Cheapest Airlines")
            st.bar_chart(avg_prices.head(5))

    with tab3:
        st.subheader("Price Outliers ($500+ over competitors)")
        df['price_gap'] = df['swiss_price'] - df['lowest_competitor_price']
        outliers = df[df['price_gap'] > 500]
        st.dataframe(outliers[['date', 'origin', 'destination', 'flight_number', 'swiss_price', 'lowest_competitor_price', 'price_gap']])

        st.subheader("Flight Duration vs SWISS Ticket Price")
        fig, ax = plt.subplots()
        ax.scatter(df['duration_minutes'], df['swiss_price'])
        ax.set_xlabel('Flight Duration (minutes)')
        ax.set_ylabel('SWISS Ticket Price (USD)')
        ax.set_title('Flight Duration vs SWISS Ticket Price')
        st.pyplot(fig)

        st.subheader("Monthly Average SWISS Price")
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly_avg = df.groupby('month')['swiss_price'].mean()
        st.line_chart(monthly_avg)

    st.download_button(
        label="Download Filtered Data as CSV",
        data=df.to_csv(index=False),
        file_name='filtered_flight_data.csv',
        mime='text/csv'
    )

folder_path = r"C:\Users\kryst\OneDrive\Desktop\airlines"

# MAIN
if __name__ == "__main__":
    dataframe = load_json_files(folder_path)

    if not dataframe.empty:
        basic_eda(dataframe)
        # OPTIONAL: Uncomment to run Streamlit dashboard automatically
        # streamlit_dashboard(dataframe)
    else:
        print("No data loaded. Check your folder path or files.")

# OPTIONAL: Run dashboard after collection automatically
    # streamlit_dashboard(dataframe)

# Save script (i.e., as swiss_dashboard.py)

# streamlit run swiss_dashboard.py

# [theme]
# primaryColor = "#d71920"  # Swiss Red
# backgroundColor = "#ffffff"
# secondaryBackgroundColor = "#f2f2f2"
# textColor = "#000000"
# font = "sans serif"