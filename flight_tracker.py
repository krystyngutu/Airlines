import requests
import pandas as pd
from datetime import datetime, timedelta
import json

# API keys
hasdata_API = '193e9ef1-70f7-4c68-8c33-11b4e155916a'

# Airport codes
nyc_airports = ['JFK', 'EWR', 'LGA']
swiss_airports = ['ZRH', 'GVA', 'BSL']

# Define desired date [range]
start_date = datetime.today() + timedelta(days=1)
end_date = start_date + timedelta(days=30)

# Define function to retrieve data from HasData
def fetch_hasdata_prices(origin, destination, outbound_date):
    url = 'https://api.hasdata.com/scrape/google/flights'
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': hasdata_API
    }

    params = {
        'departureId': origin,
        'arrivalId': destination,
        'outboundDate': outbound_date.strftime("%Y-%m-%d")
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()  # Will give error if API call fails

    return response.json()

# Main data collection loop
def collect_data():
    current_date = start_date
    all_data = []

    while current_date <= end_date:
        for origin in nyc_airports:
            for destination in swiss_airports:
                try:
                    price_data = fetch_hasdata_prices(origin, destination, current_date)
                    with open(f"C:\\Users\\kryst\\OneDrive\\Desktop\\airlines\\{current_date.strftime('%Y_%m_%d')}_{origin}_{destination}-flights.json", "w") as outputFile:
                        json.dump(price_data, outputFile)
                        outputFile.flush()
                except Exception as e:
                    print(f"Error fetching price data: {e}")
                    price_data = {}

        current_date += timedelta(days=1)

    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_data)
    return df

# Save the final dataframe
if __name__ == "__main__":
    print("Starting data collection...")

    try:
        dataframe = collect_data()
    except Exception as e:
        print(f"Failed during collection: {e}")
        exit()

    if dataframe.empty:
        print("No flight data was collected. Please check your date range or API settings.")
    else:
        print(f"Collected {len(dataframe)} flight records!")
        dataframe.to_csv("swiss_vs_competitors_flights.csv", index=False)
        print("Data saved to 'swiss_vs_competitors_flights.csv'!")