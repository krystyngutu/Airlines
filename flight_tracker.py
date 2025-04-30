import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import os

# API key
hasdata_API = '193e9ef1-70f7-4c68-8c33-11b4e155916a'

# Airport codes
nyc_airports = ['JFK', 'EWR', 'LGA']
swiss_airports = ['ZRH', 'GVA', 'BSL']

# Define desired date [range]
start_date = datetime.today() + timedelta(days=9)
end_date = start_date + timedelta(days=68)

# Paths
output_folder = r"C:\Users\kryst\OneDrive\Desktop\airlines"
output_csv = os.path.join(output_folder, "all_flights.csv")
os.makedirs(output_folder, exist_ok=True)

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
def collect_flight_jsons():
    current_date = start_date

    while current_date <= end_date:
        for origin in nyc_airports:
            for destination in swiss_airports:
                try:
                    print(f"Fetching {origin} -> {destination} on {current_date.strftime('%Y-%m-%d')}")
                    price_data = fetch_hasdata_prices(origin, destination, current_date)
                    
                    filename = f"{current_date.strftime('%Y_%m_%d')}_{origin}_{destination}_flights.json"
                    filepath = os.path.join(output_folder, filename)
                    with open(filepath, "w") as outputFile:
                        json.dump(price_data, outputFile)
                        outputFile.flush()
                except Exception as e:
                    print(f"Error fetching price data: {e}")

        current_date += timedelta(days=1)

# Function to process JSON files and transform data into CSV file 
def process_flight_data(output_folder_path):
    all_data = []

    # Loop through all JSON files in the folder
    for filename in os.listdir(output_folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(output_folder_path, filename)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    flights = data.get('bestFlights', []) + data.get('otherFlights', [])
                    
                    for flight in flights:
                        for leg in flight.get('flights', []):
                            record = {
                                'departureAirportID': leg.get('departureAirport', {}).get('id'),
                                'departureAirportName': leg.get('departureAirport', {}).get('name'),
                                'departureTime': leg.get('departureAirport', {}).get('time'),
                                'arrivalAirportID': leg.get('arrivalAirport', {}).get('id'),
                                'arrivalAirportName': leg.get('arrivalAirport', {}).get('name'),
                                'arrivalAirportTime': leg.get('arrivalAirport', {}).get('time'),
                                'airline': leg.get('airline'),
                                'flightNumber': leg.get('flightNumber'),
                                'airplane': leg.get('airplane'),
                                'travelClass': leg.get('travelClass'),
                                'legroom': leg.get('legroom'),
                                'extensions': ", ".join(leg.get('extensions', [])),
                                'durationTime': leg.get('duration'),
                                'price': flight.get('price'),
                                'tripType': flight.get('type'),
                                'totalTripDuration': flight.get('totalDuration'),
                                'carbonEmissionsThisFlight': flight.get('carbonEmissions', {}).get('thisFlight'),
                                'carbonEmissionsTypicalRoute': flight.get('carbonEmissions', {}).get('typicalForThisRoute'),
                                'carbonDifferencePercent': flight.get('carbonEmissions', {}).get('differencePercent'),
                                'layovers': ", ".join([layover.get('name', '') for layover in flight.get('layovers', [])]),
                                'departureToken': flight.get('departureToken'),
                                'metadataURL': data.get('requestMetadata', {}).get('url'),
                                'metadataHTML': data.get('requestMetadata', {}).get('html')
                            }
                            all_data.append(record)

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    return pd.DataFrame(all_data)

# Full pipeline
if __name__ == "__main__":
    print("Starting data collection...")

    try:
        collect_flight_jsons()
        df = process_flight_data(output_folder)
    except Exception as e:
        print(f"Failed: {e}")
        exit()

    if df.empty:
        print("No flight data collected. Please check your API setup or parameters.")
    else:
        df.to_csv(output_csv, index=False)
        print(f"Collected {len(df)} flight records.")
        print(f"Data saved to '{output_csv}'")
