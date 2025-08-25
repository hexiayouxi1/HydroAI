import os
import pandas as pd
import traceback
import openmeteo_requests
import requests_cache
from retry_requests import retry
from typing import Optional
import earthaccess
import xarray as xr
import pandas as pd
import requests
from datetime import datetime, timedelta
import os

# Import all the new data fetching functions
from data_sources import (
    get_geoglows_package_data_and_plot,
    get_australia_daily_data_v2,
    get_canada_realtime_data,
    get_uk_realtime_data,
    get_india_cwc_data,
    get_usgs_realtime_data,
    get_geoglows_data_from_source
)
def run_geoglows_analysis(lat, lon, analysis_type):
    return get_geoglows_package_data_and_plot(lat, lon, analysis_type)

def get_geoglows_data_only(lat, lon, analysis_type): # <-- ADD THIS WRAPPER
    return get_geoglows_data_only(lat, lon, analysis_type)
def fetch_external_data(source: str, identifier: str, days: int = 7) -> (pd.DataFrame, pd.DataFrame):
    """
    Dispatcher to fetch data from various external sources.

    Args:
        source (str): The data source ('USGS', 'GEOGLOWS', 'CANADA', 'AUSTRALIA', 'UK', 'INDIA').
        identifier (str): The station ID or "lat,lon" for GEOGloWS.
        days (int): The number of past days of data to fetch.

    Returns:
        A tuple of two pandas DataFrames: (primary_df, secondary_df).
        secondary_df is only used by GEOGloWS for forecasts.
    """
    source = source.upper()
    print(f"--- Dispatching data request for source: {source}, identifier: {identifier} ---")

    if source == 'USGS':
        df = get_usgs_realtime_data(site_id=identifier, days=days)
        return df, None

    elif source == 'CANADA':
        df = get_canada_realtime_data(station_id=identifier, days=days)
        return df, None

    elif source == 'AUSTRALIA':
        df = get_australia_daily_data_v2(station_number=identifier, days=days)
        return df, None

    elif source == 'UK':
        # UK limit is number of readings, not days. Let's approximate.
        # Assuming 15-min data, 96 readings/day.
        limit = days * 96
        df = get_uk_realtime_data(station_id=identifier, limit=limit)
        return df, None

    elif source == 'INDIA':
        # India function does not take an identifier or days
        df = get_india_cwc_data()
        return df, None

    else:
        print(f" > Error: Unknown data source '{source}'.")
        return pd.DataFrame(), None


# --- The rest of data_agent.py remains the same ---

class WeatherProvider:
    """Fetches future weather forecasts from the Open-Meteo API."""
    def __init__(self, lat: float, lon: float):
        self.lat, self.lon = lat, lon
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)
        print(f"WeatherProvider initialized for Lat={lat}, Lon={lon}.")

    def get_future_forecast(self, prediction_days: int) -> Optional[pd.DataFrame]:
        """Fetches and returns a DataFrame of future weather."""
        print(f"Fetching {prediction_days}-day forecast from Open-Meteo...")
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": self.lat, "longitude": self.lon,
                "daily": ["temperature_2m_mean", "precipitation_sum"],
                "forecast_days": prediction_days
            }
            responses = self.client.weather_api(url, params=params)
            response = responses[0]
            daily = response.Daily()

            if daily.VariablesLength() < 2:
                raise ValueError("API did not return expected weather variables.")

            time_index = pd.to_datetime(daily.Time(), unit="s", utc=True).tz_convert(None)

            df_forecast = pd.DataFrame(data={
                "date": time_index,
                "temp_c": daily.Variables(0).ValuesAsNumpy(),
                "precip_mm": daily.Variables(1).ValuesAsNumpy()
            }).set_index('date')

            print(f"✅ Successfully fetched data for {len(df_forecast)} days.")
            return df_forecast
        except Exception as e:
            print(f"❌ Error fetching from Open-Meteo: {e}")
            traceback.print_exc()
            return None

def load_and_prepare_data(filepath: str, date_col: str, flow_col: str, precip_col: str, temp_col: str) -> pd.DataFrame:
    """Loads a user-provided CSV and standardizes it."""
    df = pd.read_csv(filepath)
    rename_map = {flow_col: 'streamflow_m3s', precip_col: 'precip_mm', temp_col: 'temp_c'}
    df.rename(columns=rename_map, inplace=True)
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    for col in ['streamflow_m3s', 'precip_mm', 'temp_c']:
        if col not in df.columns: df[col] = pd.NA
    return df.sort_index().asfreq('D')

def process_camels_data(camels_root_path: str, basin_id: str) -> str:
    """
    Processes CAMELS data for a specific basin and creates a single CSV file.
    This function is part of the Data Agent's responsibilities.

    Args:
        camels_root_path: The root directory of the CAMELS dataset.
        basin_id: The 8-digit USGS basin ID string (e.g., '01013500').

    Returns:
        A string indicating the success or failure of the operation.
    """
    if not camels_root_path or not basin_id:
        return "❌ Error: Please provide both the CAMELS root path and a Basin ID."

    try:
        # --- 1. Define file paths ---
        streamflow_dir = os.path.join(camels_root_path, 'basin_dataset_public_v1p2', 'usgs_streamflow')
        streamflow_file = os.path.join(streamflow_dir, f"{basin_id}_streamflow_qc.txt")

        forcing_root_dir = os.path.join(camels_root_path, 'basin_dataset_public_v1p2', 'basin_mean_forcing', 'daymet')
        basin_id_prefix = basin_id[:2]
        forcing_dir = os.path.join(forcing_root_dir, basin_id_prefix)
        forcing_file = os.path.join(forcing_dir, f"{basin_id}_lump_cida_forcing_leap.txt")

        if not os.path.exists(streamflow_file):
            return f"❌ Error: Streamflow file not found at {streamflow_file}"
        if not os.path.exists(forcing_file):
            return f"❌ Error: Forcing data file not found at {forcing_file}"

        # --- 2. Read and process streamflow ---
        df_flow = pd.read_csv(
            streamflow_file,
            sep=r'\s+',
            header=None,
            names=['basin', 'year', 'month', 'day', 'flow_cfs', 'qc_flag']
        )
        df_flow['date'] = pd.to_datetime(df_flow[['year', 'month', 'day']])
        df_flow['streamflow_m3s'] = df_flow['flow_cfs'] * 0.0283168
        df_flow.set_index('date', inplace=True)
        df_flow = df_flow[['streamflow_m3s']]

        # --- 3. Read and process meteorological forcing data ---
        forcing_column_names = [
            'Year', 'Month', 'Day', 'Hr', 'dayl(s)', 'prcp(mm/day)',
            'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)'
        ]
        df_forcing = pd.read_csv(
            forcing_file,
            sep=r'\s+',
            skiprows=4,
            header=None,
            names=forcing_column_names
        )

        df_forcing['date'] = pd.to_datetime(df_forcing[['Year', 'Month', 'Day']])
        df_forcing['temp_c'] = (df_forcing['tmax(C)'] + df_forcing['tmin(C)']) / 2
        df_forcing.rename(columns={'prcp(mm/day)': 'precip_mm'}, inplace=True)
        df_forcing.set_index('date', inplace=True)
        df_forcing = df_forcing[['precip_mm', 'temp_c']]

        # --- 4. Merge data ---
        df_merged = df_flow.join(df_forcing, how='inner').reset_index()

        # --- 5. Save as CSV ---
        output_filename = 'camels_processed_data.csv'
        df_merged.to_csv(output_filename, index=False)
        return f"✅ Success! Created '{output_filename}' with {len(df_merged)} rows. You can now load this file."

    except Exception as e:
        return f"❌ An unexpected error occurred: {e}\n{traceback.format_exc()}"

def download_imerg_csv(bounding_box, start_date, end_date, output_csv):
    """
    Download GPM IMERG data for the specified bounding box and time range,
    compute the average precipitation over the region for each time step,
    and save it to a CSV file.

    Parameters:
    - bounding_box: tuple (lon_min, lat_min, lon_max, lat_max)
    - start_date: str, format 'YYYY-MM-DD'
    - end_date: str, format 'YYYY-MM-DD'
    - output_csv: str, path to save the CSV file

    Returns:
    - str: Path to the saved CSV file or error message
    """
    try:
        # 认证 Earthdata Login
        auth = earthaccess.login(strategy="netrc")
        if not auth.authenticated:
            return "❌ Error: Earthdata Login authentication failed. Please configure .netrc with valid credentials."

        # 搜索 IMERG Final Run 每日数据
        results = earthaccess.search_data(
            short_name="GPM_3IMERGDF",
            version="07",  # 使用最新版本
            cloud_hosted=True,
            temporal=(start_date, end_date),
            bounding_box=bounding_box
        )

        if not results:
            return f"❌ Error: No data found for the specified region and time range."

        # 下载数据
        data_dir = "imerg_data"
        os.makedirs(data_dir, exist_ok=True)
        files = earthaccess.download(results, data_dir)

        if not files:
            return "❌ Error: Failed to download GPM IMERG data."

        # 使用 xarray 打开所有文件
        ds = xr.open_mfdataset(files, combine='by_coords')

        # 提取降水数据
        if 'precipitationCal' not in ds.variables:
            return "❌ Error: Precipitation data ('precipitationCal') not found in dataset."
        precip = ds['precipitationCal']

        # 选择区域
        lon_min, lat_min, lon_max, lat_max = bounding_box
        precip_region = precip.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

        # 计算区域平均降水
        precip_mean = precip_region.mean(dim=['lon', 'lat'])

        # 转换为 DataFrame
        df = precip_mean.to_dataframe(name='precipitation')
        df.reset_index(inplace=True)
        df['date'] = pd.to_datetime(df['time']).dt.date
        df.drop('time', axis=1, inplace=True)

        # 保存为 CSV
        df.to_csv(output_csv, index=False)

        print(f"✅ Data saved to {output_csv}")
        return f"✅ Data saved to {output_csv}"

    except Exception as e:
        error_message = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return error_message