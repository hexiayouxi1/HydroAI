import requests
import pandas as pd
import io
import time
import re
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import geoglows
import traceback
from typing import Optional
# --- GEOGloWS Global Model ---
def get_geoglows_data_from_source(lat: float, lon: float, data_type: str) -> Optional[pd.DataFrame]:
    """
    Uses the 'geoglows' package to fetch data and returns it as a DataFrame.
    """
    print(f"Fetching GEOGloWS '{data_type}' data for coords: Lat={lat}, Lon={lon}")
    try:
        with requests.Session() as s:
            res = s.get('https://geoglows.ecmwf.int/api/v2/getreachid', params={'lat': lat, 'lon': lon})
            res.raise_for_status()
            river_id = res.json()['reach_id']

        df = None
        if data_type == 'forecast_stats':
            df = geoglows.data.forecast_stats(river_id)
        else:  # Add other types if needed later
            return None

        df.index = pd.to_datetime(df.index)
        df[df < 0] = 0
        return df
    except Exception:
        return None
def get_geoglows_package_data_and_plot(lat: float, lon: float, data_type: str) -> str:
    """
    Uses the official 'geoglows' package to fetch data and create an interactive plot.
    Includes robust data cleaning and detailed error logging for the plotting step.
    """
    print(f"Fetching GEOGloWS '{data_type}' plot for coords: Lat={lat}, Lon={lon}")

    # ... (Coordinate validation and River ID lookup steps are fine, no changes needed) ...
    # Step 1: Get the River ID (comid) from lat/lon...
    river_id = None
    try:
        with requests.Session() as s:
            res = s.get('https://geoglows.ecmwf.int/api/v2/getriverid', params={'lat': lat, 'lon': lon})
            res.raise_for_status()
            response_json = res.json()
            if 'river_id' not in response_json or response_json['river_id'] is None:
                 raise ValueError("API response did not contain a valid 'reach_id'.")
            river_id = response_json['river_id']
            print(f"  > Found River ID (COMID): {river_id}")
    except Exception as e:
        error_msg = f"Failed to retrieve River ID. Location may be too far from a river. Error: {e}"
        print(f"  > {error_msg}")
        return f"<p style='color:red;'>{error_msg}</p>"

    # Step 2: Use the geoglows package to fetch data
    df = None
    try:
        if data_type == 'retrospective':
            df = geoglows.data.retrospective(river_id)
        elif data_type == 'forecast_stats':
            df = geoglows.data.forecast_stats(river_id)
        elif data_type == 'forecast_ensembles':
            df = geoglows.data.forecast_ensembles(river_id)
        elif data_type == 'forecast':
            df = geoglows.data.forecast(river_id)
        else:
            raise ValueError(f"Invalid data_type '{data_type}' requested.")
        print(f"  > Successfully fetched data of shape: {df.shape}")
    except Exception as e:
        error_msg = f"Failed to fetch GEOGloWS data using the package. Error: {e}"
        print(f"  > {error_msg}")
        return f"<p style='color:red;'>{error_msg}</p>"


    # --- FIX: IMPROVED DATA CLEANING AND ERROR REPORTING FOR PLOTTING ---
    fig = None
    try:
        # Step 3a: Robust Data Cleaning
        df.index = pd.to_datetime(df.index)
        # Ensure all data columns are numeric, converting non-numeric values to NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True) # Drop rows with any conversion errors
        df[df < 0] = 0 # Set negative flows to zero

        if df.empty:
            raise ValueError("Data became empty after cleaning. Check the raw data source.")

        print("  > Data cleaned successfully. Proceeding to plot.")

        # Step 3b: Generate the plot
        plot_title = f"{data_type.replace('_', ' ').title()} for River ID: {river_id}<br>Lat: {lat}, Lon: {lon}"
        if data_type == 'retrospective':
            fig = geoglows.plots.retrospective(df, plot_titles=[plot_title])
        elif data_type == 'forecast_stats':
            fig = geoglows.plots.forecast_stats(df, plot_titles=[plot_title])
        elif data_type == 'forecast_ensembles':
            fig = geoglows.plots.forecast_ensembles(df, plot_titles=[plot_title])
        elif data_type == 'forecast':
            fig = geoglows.plots.forecast(df, plot_titles=[plot_title])
        html_str = fig.to_html(full_html=True, include_plotlyjs='cdn')
        filename = f"geoglows_{data_type}_{river_id}.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_str)
        print(f"  > Plot saved to {filename}")
        # Step 3c: Convert the plotly figure to an HTML string for Gradio
        return html_str

    except Exception as e:
        # Provide a much more detailed error message
        detailed_error = traceback.format_exc()
        error_msg = (f"<b>Plotting Failed:</b> Successfully fetched data, but encountered an error during plot generation.<br>"
                     f"<b>Data Shape After Cleaning:</b> {df.shape if df is not None else 'N/A'}<br>"
                     f"<b>Error Type:</b> {type(e).__name__}<br>"
                     f"<b>Details:</b> {e}<br><br>"
                     f"<pre>{detailed_error}</pre>") # Use <pre> for formatted traceback
        print(f"  > PLOTTING ERROR: {e}\n{detailed_error}")
        return f"<p style='color:red;'>{error_msg}</p>"
# --- Australia (BoM) ---
def get_australia_daily_data_v2(station_number: str, days: int = 90) -> pd.DataFrame:
    """
    Fetches daily mean discharge data for an Australian BoM station.
    """
    print(f"Discovering timeseries ID for BoM station: {station_number}...")
    api_url = "http://www.bom.gov.au/waterdata/services"
    try:
        discovery_params = {
            "service": "kisters", "type": "queryServices", "request": "getTimeseriesList",
            "station_no": station_number, "format": "json"
        }
        response = requests.get(api_url, params=discovery_params)
        response.raise_for_status()
        data = response.json()
        timeseries_id = None
        for ts in data:
            ts_name = ts.get('ts_name', '').lower()
            param_name = ts.get('parametertype_name', '').lower()
            if ('discharge' in param_name or 'flow' in param_name) and 'daily' in ts_name and 'mean' in ts_name:
                timeseries_id = ts.get('ts_id')
                print(f"  > Found matching timeseries ID: {timeseries_id}")
                break
        if not timeseries_id:
            print(f"  > Could not find a Daily Mean Discharge timeseries for station {station_number}.")
            return pd.DataFrame()

        print(f"Fetching values for timeseries ID: {timeseries_id}...")
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        fetch_params = {
            "service": "kisters", "type": "queryServices", "request": "getTimeseriesValues",
            "ts_id": timeseries_id, "from": start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "to": end_date.strftime('%Y-%m-%dT%H:%M:%SZ'), "format": "json"
        }
        response = requests.get(api_url, params=fetch_params)
        response.raise_for_status()
        data = response.json()
        if not data or not data[0].get('data'):
            print(f"  > No data returned for timeseries ID {timeseries_id}.")
            return pd.DataFrame()

        ts_data = data[0]['data']
        df = pd.DataFrame(ts_data, columns=['timestamp_ms', 'Discharge_Value', 'quality_code'])
        df['dateTime'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
        df.set_index('dateTime', inplace=True)
        df['Discharge_m3s'] = df['Discharge_Value'] * 0.011574  # ML/d to m³/s
        return df[['Discharge_m3s', 'quality_code']]

    except requests.exceptions.RequestException as e:
        print(f"  > Error during data fetching: {e}")
        return pd.DataFrame()


# --- Canada (ECCC) ---
def get_canada_realtime_data(station_id: str, days: int = 7) -> pd.DataFrame:
    """
    Fetches real-time data for a given Canadian hydrometric station by downloading a CSV.
    """
    print(f"Fetching data for ECCC station: {station_id}")
    base_url = "https://wateroffice.ec.gc.ca/services/real_time_data/csv/inline"
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # Note: The original provided function had a bug in the parameter names. Correcting them here.
    # The API expects 'stations[]', not 'station_number', for the inline CSV service.
    params = {
        "stations[]": station_id,
        "parameters[]": 6,  # 6 is for Discharge
        "start_date": start_date.strftime('%Y-%m-%d %H:%M:%S'),
        "end_date": end_date.strftime('%Y-%m-%d %H:%M:%S')
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        csv_content = response.text
        if "No data available" in csv_content or "Date," not in csv_content:
            print(f"  > No data found for station {station_id}")
            return pd.DataFrame()

        data_start_row = csv_content.find("Date,")
        df = pd.read_csv(io.StringIO(csv_content[data_start_row:]))

        df.rename(columns={'Date': 'dateTime', 'Discharge (m3/s)': 'Discharge_m3s'}, inplace=True)
        df['dateTime'] = pd.to_datetime(df['dateTime'])
        df.set_index('dateTime', inplace=True)

        return df[['Discharge_m3s']] if 'Discharge_m3s' in df.columns else pd.DataFrame()

    except requests.exceptions.RequestException as e:
        print(f"  > An error occurred: {e}")
        return pd.DataFrame()


# --- United Kingdom (EA) ---
def get_uk_realtime_data(station_id: str, limit: int = 200) -> pd.DataFrame:
    """
    Fetches the latest readings for a UK Environment Agency station.
    Note: This often provides Water Level, not Discharge.
    """
    print(f"Fetching data for UK EA station: {station_id}")
    base_url = f"https://environment.data.gov.uk/flood-monitoring/id/stations/{station_id}/readings"
    params = {"_sorted": "", "_limit": limit}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if not data.get('items'):
            print(f"  > No data found for station {station_id}")
            return pd.DataFrame()
        df = pd.DataFrame(data['items'])
        df['dateTime'] = pd.to_datetime(df['dateTime'])
        df.set_index('dateTime', inplace=True)
        df.rename(columns={'value': 'Water_Level_m'}, inplace=True)
        return df[['Water_Level_m']]
    except requests.exceptions.RequestException as e:
        print(f"  > An error occurred: {e}")
        return pd.DataFrame()


# --- India (CWC) - Web Scraping ---
def get_india_cwc_data() -> pd.DataFrame:
    """
    Fetches hydrological data by scraping the Central Water Commission (CWC) of India's website.
    This is fragile and does not take a station ID.
    """
    print("Fetching data from India's Central Water Commission (CWC)...")
    url = "http://cwc.gov.in/ffs-current-w-l"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        dfs = pd.read_html(response.content)
        if not dfs:
            print("  > Pandas could not parse any tables from the HTML.")
            return pd.DataFrame()
        df = dfs[0]
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        df.columns = [re.sub(r'\s+|\.|/', '_', str(col)).lower() for col in df.columns]
        print(f"  > Successfully scraped {len(df)} records.")
        return df
    except Exception as e:
        print(f"  > Error fetching or parsing CWC page: {e}")
        return pd.DataFrame()


# --- USA (USGS) ---
def get_usgs_realtime_data(site_id: str, days: int = 7) -> pd.DataFrame:
    """
    Fetches real-time instantaneous data for a given USGS site.
    """
    print(f"Fetching data for USGS site: {site_id}")
    base_url = "https://waterservices.usgs.gov/nwis/iv/"
    params = {
        "format": "json", "sites": site_id, "period": f"P{days}D",
        "parameterCd": "00060,00065", "siteStatus": "all"
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        all_records = []
        time_series = data['value']['timeSeries']
        for ts in time_series:
            var_code = ts['variable']['variableCode'][0]['value']
            col_name = 'Discharge_cfs' if var_code == '00060' else 'Gage_Height_ft'
            for value_entry in ts['values'][0]['value']:
                record = {
                    "dateTime": value_entry['dateTime'],
                    col_name: float(value_entry['value']),
                }
                all_records.append(record)
        if not all_records:
            print(f"  > No data found for site {site_id}")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df['dateTime'] = pd.to_datetime(df['dateTime']).dt.tz_localize(None)  # Remove timezone for consistency
        df = df.groupby('dateTime').first().reset_index()  # Handle duplicates
        df.set_index('dateTime', inplace=True)

        # Convert cfs to m³/s
        if 'Discharge_cfs' in df.columns:
            df['Discharge_m3s'] = df['Discharge_cfs'] * 0.0283168
        return df
    except requests.exceptions.RequestException as e:
        print(f"  > An error occurred: {e}")
        return pd.DataFrame()