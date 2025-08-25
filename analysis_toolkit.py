import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
import pymannkendall as mk
import shap
from typing import Dict, Union, Optional, Tuple, List
import traceback
from darts import TimeSeries
from darts.models import (
    TFTModel, NBEATSModel, BlockRNNModel, TCNModel,
    TransformerModel, LightGBMModel, VARIMA
)
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, rmse, r2_score
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime, timedelta
import os
import torch
import folium
import geopandas as gpd
from folium.plugins import HeatMap
import geoglows
import requests
import openmeteo_requests
# NSE function
def nse(actual_series: TimeSeries, pred_series: TimeSeries) -> float:
    rmse_val = rmse(actual_series, pred_series)
    actual_pd = pd.DataFrame(actual_series.values(), index=actual_series.time_index)
    pred_pd = pd.DataFrame(pred_series.values(), index=pred_series.time_index)
    df_aligned = actual_pd.join(pred_pd, how='inner', lsuffix='_actual', rsuffix='_pred').dropna()
    if df_aligned.empty:
        return -np.inf
    variance_actual = np.var(df_aligned.iloc[:, 0].values)
    if variance_actual == 0:
        return 1.0 if rmse_val == 0 else -np.inf
    return 1 - (rmse_val ** 2 / variance_actual)

# Create charts directory
if not os.path.exists('charts'):
    os.makedirs('charts')

# Plot styling
PLOT_STYLE = {
    'figure.figsize': (12, 6),
    'axes.facecolor': '#F8F9FA',
    'axes.edgecolor': '#dee2e6',
    'axes.grid': True,
    'grid.color': '#dee2e6',
    'grid.linestyle': '--',
    'axes.labelcolor': '#212529',
    'axes.titleweight': 'bold',
    'axes.titlesize': 'large',
    'axes.titlecolor': '#003366',
    'xtick.color': '#495057',
    'ytick.color': '#495057',
    'legend.frameon': True,
    'legend.facecolor': 'white',
    'legend.edgecolor': '#ced4da'
}
PRIMARY_COLOR = '#0056b3'
SECONDARY_COLOR = '#d9534f'
COVARIATE_COLOR = '#17a2b8'
VALIDATION_COLOR = '#28a745'
ACCENT_COLOR = '#ffc107'

class RunoffToolkit:
    def __init__(self, data: pd.DataFrame, station_id: str):
        self.df = data
        self.station_id = station_id
        self.df.sort_index(inplace=True)
        self.historical_data = self.df[self.df['streamflow_m3s'].notna()].copy()
        self.trained_model = None
        self.trained_model_name = None
        self.target_scaler = None
        self.covariate_scaler = None
        self.lgbm_X_train = None
        self.lgbm_y_train = None
        self.feature_names = None
        self.spatial_data = None  # Initialize spatial_data attribute
        print(f"Toolkit Initialized for station {station_id}.")
        print(f"Data available from {self.df.index.min().date()} to {self.df.index.max().date()}.")

    def get_statistics(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        if start_date or end_date:
            filtered_df = self.historical_data.copy()
            if start_date: filtered_df = filtered_df.loc[filtered_df.index >= pd.to_datetime(start_date)]
            if end_date: filtered_df = filtered_df.loc[filtered_df.index <= pd.to_datetime(end_date)]
            flow = filtered_df['streamflow_m3s'].dropna()
        else:
            flow = self.historical_data['streamflow_m3s'].dropna()
        if flow.empty: raise ValueError("No data available for the specified period.")
        stats_data = {"Statistic": ["Start Date", "End Date", "Data Points", "Mean Flow (m³/s)", "Max Flow (m³/s)",
                                    "Min Flow (m³/s)", "Std Dev (m³/s)", "50th Percentile (m³/s)"],
                      "Value": [str(flow.index.min().date()), str(flow.index.max().date()), len(flow),
                                f"{flow.mean():.2f}", f"{flow.max():.2f}", f"{flow.min():.2f}",
                                f"{flow.std():.2f}", f"{flow.median():.2f}"]}
        return pd.DataFrame(stats_data)

    def plot_hydrograph(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        data_to_plot = self.historical_data.copy()
        if start_date: data_to_plot = data_to_plot.loc[data_to_plot.index >= pd.to_datetime(start_date)]
        if end_date: data_to_plot = data_to_plot.loc[data_to_plot.index <= pd.to_datetime(end_date)]
        with plt.style.context(PLOT_STYLE):
            fig, ax = plt.subplots()
            ax.plot(data_to_plot.index, data_to_plot['streamflow_m3s'], label='Daily Streamflow', color=PRIMARY_COLOR)
            ax.set_title(f'Streamflow Hydrograph for {self.station_id}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Discharge (m³/s)')
            ax.legend()
            filepath = f'charts/hydrograph_{self.station_id}_{start_date or "start"}_{end_date or "end"}.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return filepath

    def calculate_flood_frequency(self, return_period: int) -> Dict[str, Union[str, float]]:
        annual_max = self.historical_data['streamflow_m3s'].resample('A').max().dropna()
        if len(annual_max) < 10:
            return {
                "error": f"Need at least 10 years of data for reliable analysis. Found only {len(annual_max)} years."}
        log_annual_max = np.log10(annual_max)
        params = stats.pearson3.fit(log_annual_max)
        probability = 1.0 / return_period
        q_log = stats.pearson3.ppf(1 - probability, *params)
        q = 10 ** q_log
        return {"return_period_years": return_period, "estimated_discharge_m3s": round(q, 2)}

    def detect_trend(self, data_column: str = 'streamflow_m3s', period: str = 'annual', alpha: float = 0.05) -> Dict:
        if data_column not in self.historical_data.columns:
            return {"error": f"Column '{data_column}' not found in data."}
        if period == 'annual':
            resampled_data = self.historical_data[data_column].resample('A').mean().dropna()
        elif period == 'monthly':
            resampled_data = self.historical_data[data_column].resample('M').mean().dropna()
        else:
            return {"error": "Invalid period. Choose 'annual' or 'monthly'."}
        if len(resampled_data) < 10:
            return {"error": f"Not enough data points ({len(resampled_data)}) for a reliable trend test."}
        result = mk.original_test(resampled_data, alpha=alpha)
        return {
            "data_column": data_column,
            "period": period,
            "trend": result.trend,
            "is_significant": result.h,
            "p_value": f"{result.p:.4f}",
            "sen_slope": f"{result.slope:.4f}",
            "analysis_period_start": str(resampled_data.index.min().date()),
            "analysis_period_end": str(resampled_data.index.max().date())
        }

    def run_conceptual_water_balance(
            self,
            meteo_data_csv: str,
            # Model parameters (these would typically be calibrated)
            soil_max_storage: float = 150.0,  # mm
            gw_max_storage: float = 300.0,  # mm
            surface_runoff_coeff: float = 0.4,  # Fraction of excess rain becoming runoff
            percolation_rate: float = 0.1,  # Fraction of soil moisture percolating per day
            baseflow_coeff: float = 0.05,  # Fraction of groundwater becoming baseflow per day
            pet_factor: float = 1.26  # Factor for Priestly-Taylor PET calculation
    ) -> str:
        """
        Runs a daily conceptual water balance model for a catchment.

        This model simulates key hydrological processes including evapotranspiration,
        soil moisture dynamics, groundwater recharge, surface runoff, and baseflow
        to generate a complete streamflow hydrograph from meteorological data.

        Args:
            meteo_data_csv (str): Path to CSV with daily meteorological data.
                                  Must include 'datetime', 'prcp_mm', 'tmean_c', 'srad_wm2'.
            soil_max_storage (float): Maximum water holding capacity of the soil (mm).
            gw_max_storage (float): Maximum water holding capacity of the groundwater store (mm).
            surface_runoff_coeff (float): Runoff coefficient.
            percolation_rate (float): Rate of percolation from soil to groundwater.
            baseflow_coeff (float): Rate of groundwater release as baseflow.
            pet_factor (float): Priestley-Taylor coefficient for PET calculation.

        Returns:
            str: The file path to a comprehensive plot showing the simulated water balance.
        """
        # --- 1. Load and Prepare Data ---
        try:
            df = pd.read_csv(meteo_data_csv, parse_dates=['datetime'])
            df.set_index('datetime', inplace=True)
        except FileNotFoundError:
            return "Error: Meteorological data CSV not found."

        # --- 2. Calculate Potential Evapotranspiration (PET) using Priestley-Taylor ---
        # A simple yet physically-based method
        latent_heat_vaporization = 2.45  # MJ/kg
        water_density = 1000  # kg/m^3
        psychrometric_constant = 0.066  # kPa/°C
        slope_saturation_vapor_pressure = 4098 * (
                    0.6108 * np.exp((17.27 * df['tmean_c']) / (df['tmean_c'] + 237.3))) / (df['tmean_c'] + 237.3) ** 2

        # Convert srad from W/m^2 to MJ/m^2/day
        net_radiation = df['srad_wm2'] * 0.0864 * 0.7  # Assuming albedo of 0.3

        pet = pet_factor * (
                    slope_saturation_vapor_pressure / (slope_saturation_vapor_pressure + psychrometric_constant)) * (
                          net_radiation / latent_heat_vaporization)
        df['pet_mm'] = pet

        # --- 3. Initialize Model States ---
        num_steps = len(df)
        soil_moisture = np.zeros(num_steps)
        groundwater = np.zeros(num_steps)
        actual_et = np.zeros(num_steps)
        surface_runoff = np.zeros(num_steps)
        percolation = np.zeros(num_steps)
        baseflow = np.zeros(num_steps)
        total_streamflow = np.zeros(num_steps)

        # Set initial conditions (e.g., half full)
        soil_moisture[0] = soil_max_storage / 2
        groundwater[0] = gw_max_storage / 2

        # --- 4. Run the Daily Simulation Loop ---
        for t in range(1, num_steps):
            # --- Soil Moisture Bucket ---
            # Previous day's storage
            sm_yesterday = soil_moisture[t - 1]
            precip_today = df['prcp_mm'].iloc[t]

            # Calculate Actual Evapotranspiration (AET)
            # AET is limited by PET and available water
            aet_potential = df['pet_mm'].iloc[t]
            aet_today = min(aet_potential, sm_yesterday)
            actual_et[t] = aet_today

            sm_after_et = sm_yesterday - aet_today

            # Add precipitation to soil
            sm_with_precip = sm_after_et + precip_today

            # Check for saturation and generate surface runoff
            if sm_with_precip > soil_max_storage:
                excess_water = sm_with_precip - soil_max_storage
                surface_runoff[t] = excess_water * surface_runoff_coeff
                sm_today = soil_max_storage
            else:
                surface_runoff[t] = 0
                sm_today = sm_with_precip

            # --- Percolation from Soil to Groundwater ---
            percolation_today = sm_today * percolation_rate
            sm_after_perc = sm_today - percolation_today
            soil_moisture[t] = sm_after_perc

            # --- Groundwater Bucket ---
            gw_yesterday = groundwater[t - 1]
            gw_with_perc = gw_yesterday + percolation_today

            # Calculate Baseflow
            baseflow_today = gw_with_perc * baseflow_coeff
            baseflow[t] = baseflow_today

            gw_today = gw_with_perc - baseflow_today
            # Cap groundwater at max storage (spill is lost from system)
            groundwater[t] = min(gw_today, gw_max_storage)

            # --- Total Streamflow ---
            total_streamflow[t] = surface_runoff[t] + baseflow[t]

        # --- 5. Store and Plot Results ---
        df['sim_streamflow_mm'] = total_streamflow
        df['soil_moisture_mm'] = soil_moisture
        df['baseflow_mm'] = baseflow
        df['surface_runoff_mm'] = surface_runoff
        df['actual_et_mm'] = actual_et

        fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
        # Plot 1: Precipitation and ET
        axes[0].bar(df.index, df['prcp_mm'], label='Precipitation', color='lightblue')
        axes[0].plot(df.index, df['actual_et_mm'], label='Actual ET', color='green')
        axes[0].set_ylabel('mm/day')
        axes[0].set_title('Forcings and Evapotranspiration')
        axes[0].legend()

        # Plot 2: Simulated Streamflow Components
        axes[1].plot(df.index, df['sim_streamflow_mm'], label='Total Simulated Streamflow', color='black')
        axes[1].fill_between(df.index, 0, df['baseflow_mm'], label='Baseflow', color='royalblue', alpha=0.7)
        axes[1].fill_between(df.index, df['baseflow_mm'], df['sim_streamflow_mm'], label='Surface Runoff',
                             color='darkorange', alpha=0.7)
        axes[1].set_ylabel('mm/day')
        axes[1].set_title('Simulated Streamflow Components')
        axes[1].legend()

        # Plot 3: Soil Moisture
        axes[2].plot(df.index, df['soil_moisture_mm'], label='Soil Moisture Storage', color='brown')
        axes[2].axhline(y=soil_max_storage, linestyle='--', color='red', label='Max Soil Storage')
        axes[2].set_ylabel('mm')
        axes[2].set_title('Soil Moisture Dynamics')
        axes[2].legend()

        # Plot 4: Groundwater
        axes[3].plot(df.index, groundwater, label='Groundwater Storage', color='darkblue')
        axes[3].axhline(y=gw_max_storage, linestyle='--', color='red', label='Max GW Storage')
        axes[3].set_ylabel('mm')
        axes[3].set_title('Groundwater Dynamics')
        axes[3].legend()

        plt.xlabel('Date')
        plt.tight_layout()
        output_path = f"charts/water_balance_sim_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.png"
        plt.savefig(output_path)
        plt.close(fig)

        return output_path

    def train_forecasting_model(self, model_name: str = 'TFT') -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
        try:
            print(f"Starting {model_name.upper()} model training and evaluation...")
            target_series = TimeSeries.from_series(self.historical_data['streamflow_m3s'], freq='D',
                                                   fill_missing_dates=True, fillna_value=0).astype(np.float32)
            if 'precip_mm' not in self.historical_data or 'temp_c' not in self.historical_data:
                return "❌ Error: 'precip_mm' and 'temp_c' columns are required as covariates.", None, None
            covariates = TimeSeries.from_dataframe(self.historical_data, value_cols=['precip_mm', 'temp_c'], freq='D',
                                                   fill_missing_dates=True, fillna_value=0).astype(np.float32)
            val_split_point = len(target_series) - 365 * 2
            if val_split_point < 90:
                return "❌ Error: Not enough data for training/validation split. Need > 2.5 years of data.", None, None
            train_target, val_target = target_series.split_before(val_split_point)
            input_chunk, output_chunk = 90, 30
            model_name_upper = model_name.upper()
            deep_learning_models = {'TFT', 'NBEATS', 'BLOCKRNN', 'TCN', 'TRANSFORMER'}
            pl_trainer_kwargs = {"accelerator": "auto",
                                 "callbacks": [EarlyStopping(monitor="val_loss", patience=5, mode="min")]}
            model_map = {
                'TFT': TFTModel(input_chunk_length=input_chunk, output_chunk_length=output_chunk, n_epochs=30,
                                pl_trainer_kwargs=pl_trainer_kwargs, random_state=42, add_relative_index=True),
                'NBEATS': NBEATSModel(input_chunk_length=input_chunk, output_chunk_length=output_chunk, n_epochs=30,
                                      pl_trainer_kwargs=pl_trainer_kwargs, random_state=42),
                'BLOCKRNN': BlockRNNModel(model='LSTM', input_chunk_length=input_chunk,
                                          output_chunk_length=output_chunk, n_epochs=50,
                                          pl_trainer_kwargs=pl_trainer_kwargs, random_state=42),
                'TCN': TCNModel(input_chunk_length=input_chunk, output_chunk_length=output_chunk, n_epochs=50,
                                pl_trainer_kwargs=pl_trainer_kwargs, random_state=42),
                'TRANSFORMER': TransformerModel(input_chunk_length=input_chunk, output_chunk_length=output_chunk,
                                                d_model=32, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                                                n_epochs=30, pl_trainer_kwargs=pl_trainer_kwargs, random_state=42),
                'LIGHTGBM': LightGBMModel(lags=input_chunk, lags_past_covariates=input_chunk,
                                          output_chunk_length=output_chunk, random_state=42),
                'VARIMA': VARIMA(p=5, d=1, q=2, random_state=42),
            }
            model = model_map.get(model_name_upper)
            if model is None: return f"❌ Error: Model '{model_name}' not supported.", None, None
            if model_name_upper in deep_learning_models:
                print("Using scaled data for deep learning model.")
                self.target_scaler = Scaler()
                self.covariate_scaler = Scaler()
                scaled_train_target = self.target_scaler.fit_transform(train_target)
                scaled_covariates = self.covariate_scaler.fit_transform(covariates)
                scaled_train_cov, scaled_val_cov = scaled_covariates.split_before(val_split_point)
                scaled_val_target = self.target_scaler.transform(val_target)
                model.fit(
                    series=scaled_train_target,
                    past_covariates=scaled_train_cov,
                    val_series=scaled_val_target,
                    val_past_covariates=scaled_val_cov,
                    verbose=True
                )
            else:
                print("Using original data for classical/tree-based model.")
                model.fit(
                    series=train_target,
                    past_covariates=covariates
                )
                if model_name_upper == 'LIGHTGBM':
                    self.lgbm_X_train, self.feature_names = self._build_regression_matrix(
                        train_target, covariates, list(range(1, input_chunk + 1)), list(range(1, input_chunk + 1))
                    )
                    self.lgbm_y_train = train_target.values()
            self.trained_model = model
            self.trained_model_name = model_name_upper
            print("Evaluating model on validation set...")
            val_forecast = model.predict(
                n=len(val_target),
                series=train_target,
                past_covariates=covariates
            )
            if not isinstance(val_forecast, TimeSeries):
                print("Model output was not a TimeSeries. Reconstructing from numpy array.")
                val_forecast = TimeSeries.from_times_and_values(
                    times=val_target.time_index,
                    values=val_forecast,
                    columns=val_target.columns
                )
            if model_name_upper in deep_learning_models and self.target_scaler:
                val_forecast = self.target_scaler.inverse_transform(val_forecast)
            r2 = r2_score(val_target, val_forecast)
            nse_val = nse(val_target, val_forecast)
            metrics_data = {
                'Metric': ['R-squared (R²)', 'Nash-Sutcliffe (NSE)', 'Mean Absolute Error (MAE)',
                           'Root Mean Squared Error (RMSE)'],
                'Value': [f"{r2:.4f}", f"{nse_val:.4f}", f"{mae(val_target, val_forecast):.4f}",
                          f"{rmse(val_target, val_forecast):.4f}"]
            }
            metrics_df = pd.DataFrame(metrics_data)
            plot_path = self._create_evaluation_plot(train_target, val_target, val_forecast)
            status_message = f"✅ {self.trained_model_name} trained successfully. Evaluation complete."
            return status_message, metrics_df, plot_path
        except Exception as e:
            error_message = f"❌ Training Failed: {e}"
            traceback.print_exc()
            return error_message, None, self._create_error_plot(str(e))

    def _create_evaluation_plot(self, train_series: TimeSeries, val_series: TimeSeries,
                                val_forecast: TimeSeries) -> str:
        with plt.style.context(PLOT_STYLE):
            fig, ax = plt.subplots()
            train_series[-180:].plot(ax=ax, label="Training Data (Recent)", color=PRIMARY_COLOR, lw=1.5)
            val_series.plot(ax=ax, label="Validation Data (Actual)", color=VALIDATION_COLOR, lw=2)
            val_forecast.plot(ax=ax, label=f"{self.trained_model_name} Forecast (Validation)", color=SECONDARY_COLOR,
                              linestyle='--')
            ax.set_title(f'Model Evaluation: {self.trained_model_name} on Validation Set')
            ax.set_xlabel('Date')
            ax.set_ylabel('Discharge (m³/s)')
            ax.legend()
            filepath = f'charts/evaluation_{self.station_id}_{self.trained_model_name}.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return filepath

    def predict_with_model(self, prediction_days: int, future_weather_df: pd.DataFrame) -> str:
        if self.trained_model is None: return self._create_error_plot(
            "Model not trained yet. Please run 'train_forecasting_model' first.")
        print(f"Generating a {prediction_days}-day forecast with {self.trained_model_name}...")
        historical_covariates_df = self.historical_data[['precip_mm', 'temp_c']]
        all_covariates_df = pd.concat([historical_covariates_df, future_weather_df])
        all_covariates_df = all_covariates_df[~all_covariates_df.index.duplicated(keep='first')].asfreq('D').fillna(0)
        all_past_covariates_ts = TimeSeries.from_dataframe(all_covariates_df, freq='D', fill_missing_dates=True,
                                                           fillna_value=0).astype(np.float32)
        series_for_prediction = TimeSeries.from_series(self.historical_data['streamflow_m3s'], freq='D',
                                                       fill_missing_dates=True, fillna_value=0).astype(np.float32)
        deep_learning_models = {'TFT', 'NBEATS', 'BLOCKRNN', 'TCN', 'TRANSFORMER'}
        if self.trained_model_name in deep_learning_models and self.target_scaler and self.covariate_scaler:
            scaled_series = self.target_scaler.transform(series_for_prediction)
            scaled_covariates = self.covariate_scaler.transform(all_past_covariates_ts)
            prediction = self.trained_model.predict(
                n=prediction_days,
                series=scaled_series,
                past_covariates=scaled_covariates
            )
            prediction = self.target_scaler.inverse_transform(prediction)
        else:
            prediction = self.trained_model.predict(
                n=prediction_days,
                series=series_for_prediction,
                past_covariates=all_past_covariates_ts
            )
        with plt.style.context(PLOT_STYLE):
            fig, ax1 = plt.subplots()
            series_for_prediction[-90:].plot(ax=ax1, label='Historical Runoff', color=PRIMARY_COLOR)
            prediction.plot(ax=ax1, label=f'Predicted Runoff ({self.trained_model_name})', color=SECONDARY_COLOR)
            ax1.set_title(f'Runoff Forecast for {self.station_id}')
            ax1.set_ylabel('Discharge (m³/s)')
            ax1.legend(loc='upper left')
            ax2 = ax1.twinx()
            future_precip = future_weather_df['precip_mm']
            ax2.bar(future_precip.index, future_precip.values, label='Future Precipitation', color=COVARIATE_COLOR,
                    alpha=0.5, width=0.5)
            ax2.set_ylabel('Precipitation (mm)', color=COVARIATE_COLOR)
            ax2.tick_params(axis='y', labelcolor=COVARIATE_COLOR)
            ax2.invert_yaxis()
            ax2.legend(loc='upper right')
            fig.tight_layout()
            filepath = f'charts/forecast_{self.station_id}.png'
            plt.savefig(filepath, dpi=150)
            plt.close(fig)
            return filepath

    def _build_regression_matrix(self, target_series: TimeSeries, past_covariates: TimeSeries, lags: List[int],
                                 lags_past_covariates: List[int]) -> Tuple[np.ndarray, List[str]]:
        print("Manually building feature matrix for SHAP using pure pandas...")
        try:
            target_df = target_series.pd_dataframe()
            covariates_df = past_covariates.pd_dataframe()
        except AttributeError:
            target_df = pd.DataFrame(target_series.values(), index=target_series.time_index, columns=['streamflow_m3s'])
            covariates_df = pd.DataFrame(past_covariates.values(), index=past_covariates.time_index,
                                         columns=past_covariates.components)
        full_df = pd.concat([target_df, covariates_df], axis=1)
        feature_dfs = []
        feature_names = []
        all_lags = sorted(list(set(lags + lags_past_covariates)), reverse=True)
        for lag in all_lags:
            shifted_df = full_df.shift(lag)
            shifted_cols = [f"{col}_lag{lag}" for col in full_df.columns]
            shifted_df.columns = shifted_cols
            feature_dfs.append(shifted_df)
            feature_names.extend(shifted_cols)
        X_df = pd.concat(feature_dfs, axis=1)
        y_df = target_df
        final_df = pd.concat([y_df, X_df], axis=1).dropna()
        X_matrix = final_df.iloc[:, 1:].values
        if X_matrix.shape[0] == 0:
            raise ValueError("Built matrix is empty. The time series might be too short for the given lags.")
        print(f"✅ Manually built matrix with shape: {X_matrix.shape}")
        return X_matrix, feature_names

    def explain_forecast(self) -> Union[Tuple[str, str], str]:
        print("\n--- Starting On-Demand SHAP Analysis ---")
        if self.trained_model is None:
            return self._create_error_plot("No model has been trained yet. Please train a model first.")
        print(f"Attempting SHAP for model: {self.trained_model_name}")
        deep_learning_models = {'TFT', 'NBEATS', 'BLOCKRNN', 'TCN', 'TRANSFORMER'}
        is_deep_learning = self.trained_model_name in deep_learning_models
        try:
            print("Accessing model configuration to get lags...")
            if self.trained_model_name == 'VARIMA':
                lags = list(range(1, 6))
                lags_past_covariates = list(range(1, 6))
            elif hasattr(self.trained_model, 'model_params'):
                model_params = self.trained_model.model_params
                input_chunk_length = model_params.get('input_chunk_length', 90)
                lags = list(range(1, input_chunk_length + 1))
                lags_past_covariates = list(range(1, input_chunk_length + 1))
            elif hasattr(self.trained_model, 'lags'):
                lags = self.trained_model.lags.get('target', list(range(1, 91)))
                lags_past_covariates = self.trained_model.lags.get('past', list(range(1, 91)))
                lags = lags if isinstance(lags, list) else list(range(1, max(lags) + 1))
                lags_past_covariates = lags_past_covariates if isinstance(lags_past_covariates, list) else list(
                    range(1, max(lags_past_covariates) + 1))
            else:
                lags = list(range(1, 91))
                lags_past_covariates = list(range(1, 91))
            print(f"✅ Retrieved lags: target_lags={lags[:5]}..., covariate_lags={lags_past_covariates[:5]}...")
        except Exception as e:
            return self._create_error_plot(f"SHAP Failed: Could not get lags from model. Error: {e}")
        try:
            series = TimeSeries.from_series(self.historical_data['streamflow_m3s'])
            covariates = TimeSeries.from_dataframe(self.historical_data, value_cols=['precip_mm', 'temp_c'])
            if is_deep_learning and self.target_scaler and self.covariate_scaler:
                series = self.target_scaler.transform(series)
                covariates = self.covariate_scaler.transform(covariates)
            X_train_matrix, feature_names = self._build_regression_matrix(series, covariates, lags,
                                                                          lags_past_covariates)
        except Exception as e:
            return self._create_error_plot(f"SHAP Failed during matrix construction: {e}\n{traceback.format_exc()}")
        try:
            if is_deep_learning:
                if not hasattr(self.trained_model, 'model'):
                    return self._create_error_plot("SHAP Failed: Deep learning model not accessible for explanation.")

                def model_predict(inputs):
                    inputs_ts = TimeSeries.from_values(inputs, columns=feature_names[:inputs.shape[1]])
                    if self.covariate_scaler:
                        inputs_ts = self.covariate_scaler.transform(inputs_ts)
                    with torch.no_grad():
                        output = self.trained_model.predict(
                            n=1,
                            series=TimeSeries.from_values(inputs[:, :len(lags)], columns=['streamflow_m3s']),
                            past_covariates=inputs_ts
                        )
                    return output.values()

                explainer = shap.KernelExplainer(model_predict, X_train_matrix[:50])
                shap_values = explainer.shap_values(X_train_matrix[50:100], nsamples=100)
            else:
                if self.trained_model_name == 'LIGHTGBM':
                    if not hasattr(self.trained_model, 'model') or not self.feature_names:
                        return self._create_error_plot("SHAP Failed: LightGBM model or feature names not available.")
                    lgbm_model = self.trained_model.model.estimators_[0]
                    explainer = shap.TreeExplainer(lgbm_model)
                    shap_values = explainer.shap_values(X_train_matrix)
                else:
                    def varima_predict(inputs):
                        return self.trained_model.predict(
                            n=1,
                            series=TimeSeries.from_values(inputs[:, :len(lags)], columns=['streamflow_m3s'])
                        ).values()

                    explainer = shap.KernelExplainer(varima_predict, X_train_matrix[:50])
                    shap_values = explainer.shap_values(X_train_matrix[50:100], nsamples=100)
            print("✅ SHAP values calculated successfully.")
        except Exception as e:
            return self._create_error_plot(f"SHAP Failed during value calculation: {e}\n{traceback.format_exc()}")
        try:
            summary_path = f'charts/shap_summary_{self.station_id}_{self.trained_model_name}.png'
            dep_path = f'charts/shap_dependence_{self.station_id}_{self.trained_model_name}.png'
            shap.summary_plot(shap_values, features=X_train_matrix, feature_names=feature_names, show=False,
                              max_display=10)
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            plt.close()
            shap.dependence_plot(0, shap_values, X_train_matrix, feature_names=feature_names, show=False)
            plt.savefig(dep_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  -> Saved summary plot to {summary_path}")
            print(f"  -> Saved dependence plot to {dep_path}")
            return summary_path, dep_path
        except Exception as e:
            return self._create_error_plot(f"SHAP plot generation failed: {e}\n{traceback.format_exc()}")

    def separate_hydrograph_components(self, event_date: str, c_new: float, c_old: float) -> Union[str, Dict]:
        if 'tracer_concentration' not in self.df.columns:
            return self._create_error_plot("Hydrograph separation requires a 'tracer_concentration' column.")
        try:
            event_datetime = pd.to_datetime(event_date)
            window_start = event_datetime - timedelta(days=3)
            window_end = event_datetime + timedelta(days=5)
            event_data = self.df.loc[window_start:window_end].copy()
            baseflow = event_data['streamflow_m3s'].iloc[0]
            event_data['quickflow'] = (event_data['streamflow_m3s'] - baseflow).clip(lower=0)
            if event_data['quickflow'].sum() == 0:
                return self._create_error_plot("No significant storm event found around the given date.")
            if c_new == c_old:
                return self._create_error_plot("Concentration of new and old water cannot be the same.")
            fraction_new_water = (event_data['tracer_concentration'] - c_old) / (c_new - c_old)
            fraction_new_water = fraction_new_water.clip(0, 1)
            event_data['new_water_flow'] = event_data['streamflow_m3s'] * fraction_new_water
            event_data['old_water_flow'] = event_data['streamflow_m3s'] * (1 - fraction_new_water)
            total_volume = event_data['streamflow_m3s'].sum() * 24 * 3600
            new_water_volume = event_data['new_water_flow'].sum() * 24 * 3600
            old_water_volume = event_data['old_water_flow'].sum() * 24 * 3600
            with plt.style.context(PLOT_STYLE):
                fig, ax = plt.subplots()
                ax.stackplot(event_data.index, event_data['new_water_flow'], event_data['old_water_flow'],
                             labels=[f'New Water (Precipitation)', f'Old Water (Pre-event)'],
                             colors=[COVARIATE_COLOR, PRIMARY_COLOR], alpha=0.7)
                ax.plot(event_data.index, event_data['streamflow_m3s'], label='Total Streamflow', color='black', lw=2)
                ax.set_title(f'Hydrograph Separation for Event on {event_date}')
                ax.set_ylabel('Discharge (m³/s)')
                ax.set_xlabel('Date')
                ax.legend(loc='upper left')
                filepath = f'charts/hydrograph_separation_{self.station_id}.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close(fig)
            return {
                "plot_path": filepath,
                "total_volume_m3": f"{total_volume:,.0f}",
                "new_water_volume_m3": f"{new_water_volume:,.0f}",
                "old_water_volume_m3": f"{old_water_volume:,.0f}",
                "new_water_contribution_percent": f"{100 * new_water_volume / total_volume:.1f}%"
            }
        except Exception as e:
            return self._create_error_plot(f"Separation failed: {e}")

    def get_satellite_precipitation(self, region: str, start_date: str, end_date: str) -> str:
        print(f"MOCK: Fetching GPM satellite precipitation for '{region}'...")
        try:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            n_days = len(dates)
            precip_data = np.random.gamma(0.5, 1.5, n_days)
            precip_data[np.random.rand(n_days) < 0.75] = 0
            # Add spatial coordinates for mapping
            lat, lon = self._get_region_coordinates(region)
            n_points = 100
            lats = np.random.uniform(lat - 0.5, lat + 0.5, n_points)
            lons = np.random.uniform(lon - 0.5, lon + 0.5, n_points)
            values = np.random.gamma(0.5, 1.5, n_points)
            spatial_df = pd.DataFrame({
                'latitude': lats,
                'longitude': lons,
                'value': values
            }, index=[dates[0]] * n_points)
            self.spatial_data = spatial_df
            with plt.style.context(PLOT_STYLE):
                fig, ax = plt.subplots()
                ax.bar(dates, precip_data, color=COVARIATE_COLOR, width=1.0)
                ax.set_title(f"MOCK: Daily Satellite Precipitation (GPM) for {region}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Precipitation (mm/day)")
                ax.set_ylim(bottom=0)
                filepath = f'charts/mock_gpm_{region.replace(" ", "_")}.png'
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close(fig)
            return filepath
        except Exception as e:
            return self._create_error_plot(f"Failed to generate mock precipitation data: {e}")

    def calculate_regional_et(self, region: str, model: str = 'MOD16') -> str:
        print(f"MOCK: Calculating regional ET for '{region}' using model '{model}'...")
        dates = pd.date_range(start='2018-01-01', end='2022-12-31', freq='M')
        seasonal_cycle = 35 * (1 - np.cos(2 * np.pi * dates.month / 12)) + 15
        noise = np.random.normal(0, 5, len(dates))
        et_data = seasonal_cycle + noise
        # Add spatial coordinates for mapping
        lat, lon = self._get_region_coordinates(region)
        n_points = 100
        lats = np.random.uniform(lat - 0.5, lat + 0.5, n_points)
        lons = np.random.uniform(lon - 0.5, lon + 0.5, n_points)
        values = np.random.normal(30, 5, n_points)
        spatial_df = pd.DataFrame({
            'latitude': lats,
            'longitude': lons,
            'value': values
        }, index=[dates[0]] * n_points)
        self.spatial_data = spatial_df
        with plt.style.context(PLOT_STYLE):
            fig, ax = plt.subplots()
            ax.plot(dates, et_data, color=VALIDATION_COLOR, marker='o', linestyle='-')
            ax.set_title(f"MOCK: Monthly Regional Evapotranspiration ({model}) for {region}")
            ax.set_xlabel("Date")
            ax.set_ylabel("ET (mm/month)")
            ax.set_ylim(bottom=0)
            filepath = f'charts/mock_et_{region.replace(" ", "_")}.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
        return filepath

    def get_water_storage_anomaly(self, region: str) -> str:
        print(f"MOCK: Fetching GRACE water storage anomaly for '{region}'...")
        dates = pd.date_range(start='2002-04-01', end='2022-12-31', freq='M')
        trend = -np.linspace(0, 25, len(dates))
        seasonal = 5 * np.sin(2 * np.pi * (dates.month - 3) / 12)
        noise = np.random.normal(0, 2.5, len(dates))
        data = trend + seasonal + noise
        # Add spatial coordinates for mapping
        lat, lon = self._get_region_coordinates(region)
        n_points = 100
        lats = np.random.uniform(lat - 0.5, lat + 0.5, n_points)
        lons = np.random.uniform(lon - 0.5, lon + 0.5, n_points)
        values = np.random.normal(0, 2.5, n_points)
        spatial_df = pd.DataFrame({
            'latitude': lats,
            'longitude': lons,
            'value': values
        }, index=[dates[0]] * n_points)
        self.spatial_data = spatial_df
        with plt.style.context(PLOT_STYLE):
            fig, ax = plt.subplots()
            ax.plot(dates, data, label='TWS Anomaly', color=PRIMARY_COLOR, lw=2)
            ax.axhline(0, color='black', linestyle='--', lw=1, label='2002-2009 Baseline')
            ax.set_title(f'MOCK: Terrestrial Water Storage Anomaly for {region}')
            ax.set_ylabel('Equivalent Water Height Anomaly (cm)')
            ax.set_xlabel('Date')
            ax.legend()
            filepath = f'charts/mock_grace_{region.replace(" ", "_")}.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
        return filepath

    def map_surface_water_dynamics(self, water_body_name: str) -> str:
        print(f"MOCK: Mapping surface water dynamics for '{water_body_name}'...")
        years = np.arange(2000, 2024)
        start_area = 550 + np.random.uniform(-20, 20)
        area = start_area - np.linspace(0, 200, len(years)) + np.random.normal(0, 15, len(years))
        # Add spatial coordinates for mapping
        lat, lon = self._get_region_coordinates(water_body_name)
        n_points = 50
        lats = np.random.uniform(lat - 0.2, lat + 0.2, n_points)
        lons = np.random.uniform(lon - 0.2, lon + 0.2, n_points)
        values = np.random.normal(area[-1], 10, n_points)
        self.spatial_data = pd.DataFrame({
            'latitude': lats,
            'longitude': lons,
            'value': values
        }, index=[pd.Timestamp('2023-01-01')] * n_points)
        with plt.style.context(PLOT_STYLE):
            fig, ax = plt.subplots()
            ax.bar(years, area, color=ACCENT_COLOR, edgecolor='grey')
            z = np.polyfit(years, area, 1)
            p = np.poly1d(z)
            ax.plot(years, p(years), "r--", label=f"Trendline ({-z[0]:.1f} km²/year)")
            ax.set_title(f'MOCK: Annual Surface Area Change for {water_body_name}')
            ax.set_ylabel('Surface Area (km²)')
            ax.set_xlabel('Year')
            ax.legend()
            ax.set_xticks(years[::2])
            plt.xticks(rotation=45)
            filepath = f'charts/mock_swot_{water_body_name.replace(" ", "_")}.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
        return filepath

    def _get_region_coordinates(self, region: str) -> Tuple[float, float]:
        """Return approximate (lat, lon) for a region. In real use, this would query a geospatial database."""
        region_coords = {
            "CAMELS_Basin": (39.0, -105.0),  # Example: Colorado, USA
            "Test_Region": (40.0, -100.0),
            "United States": (37.0, -95.0),  # Approximate center of the US
            "Nile Basin": (0.0, 30.0)
        }
        return region_coords.get(region, (39.0, -105.0))

    def _generate_mock_grid_analysis(self, region: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate mock grid analysis results (e.g., NSE scores) for a region."""
        lat, lon = self._get_region_coordinates(region)
        n_points = 100
        lats = np.random.uniform(lat - 0.5, lat + 0.5, n_points)
        lons = np.random.uniform(lon - 0.5, lon + 0.5, n_points)
        values = np.random.uniform(0.5, 0.95, n_points)  # Mock NSE scores
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        spatial_df = pd.DataFrame({
            'latitude': lats,
            'longitude': lons,
            'value': values
        }, index=[dates[0]] * n_points)
        return spatial_df

    def display_map(self, data_type: str, region: str, start_date: str, end_date: str) -> str:
        try:
            print(f"Generating map for {data_type} in {region} from {start_date} to {end_date}...")
            valid_data_types = ['precipitation', 'et', 'water_storage', 'grid_analysis']
            if data_type not in valid_data_types:
                raise ValueError(f"Invalid data_type: {data_type}. Choose from {valid_data_types}.")

            # Call the corresponding data fetching method
            if data_type == 'precipitation':
                self.get_satellite_precipitation(region, start_date, end_date)
            elif data_type == 'et':
                self.calculate_regional_et(region, start_date=start_date, end_date=end_date)
            elif data_type == 'water_storage':
                self.get_water_storage_anomaly(region)
            elif data_type == 'grid_analysis':
                self.spatial_data = self._generate_mock_grid_analysis(region, start_date, end_date)

            # Check if spatial_data is set and not empty
            if self.spatial_data is None:
                raise ValueError(f"No spatial data available for {data_type} in {region}.")
            elif self.spatial_data.empty:
                raise ValueError(f"Spatial data is empty for {data_type} in {region}.")

            # Create Folium map
            lat, lon = self._get_region_coordinates(region)
            m = folium.Map(location=[lat, lon], zoom_start=8, tiles='OpenStreetMap')

            # Prepare data for visualization
            spatial_data = self.spatial_data.reset_index()
            if 'value' not in spatial_data.columns:
                raise ValueError(f"Spatial data missing 'value' column for {data_type}.")

            # Normalize values for heatmap
            values = spatial_data['value'].values
            if values.max() > values.min():
                weights = (values - values.min()) / (values.max() - values.min())
            else:
                weights = np.ones_like(values)

            # Add heatmap or markers based on data density
            if len(spatial_data) > 50:
                heat_data = [[row['latitude'], row['longitude'], w] for _, row, w in
                             zip(spatial_data.index, spatial_data.to_dict('records'), weights)]
                HeatMap(heat_data, radius=15, blur=20).add_to(m)
            else:
                for _, row in spatial_data.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=5,
                        popup=f"Value: {row['value']:.2f}",
                        color=PRIMARY_COLOR,
                        fill=True,
                        fill_color=PRIMARY_COLOR,
                        fill_opacity=0.7
                    ).add_to(m)

            # Add title (bottom-right corner, font size 12pt)
            title = f"{data_type.capitalize()} for {region} ({start_date} to {end_date})"
            title_html = f'''
                <div style="
                    position: fixed; 
                    bottom: 10px; 
                    right: 10px; 
                    z-index: 1000; 
                    font-size: 12pt; 
                    color: #003366; 
                    font-weight: bold;
                    background-color: rgba(255, 255, 255, 0.8);
                    padding: 5px;
                    border-radius: 3px;
                ">{title}</div>
            '''
            m.get_root().html.add_child(folium.Element(title_html))

            # Save map
            filepath = f'charts/map_{data_type}_{region.replace(" ", "_")}_{pd.Timestamp.now().strftime("%H%M%S")}.html'
            m.save(filepath)
            print(f"✅ Map saved to {filepath}")
            return filepath

        except Exception as e:
            error_html = f"""
            <div style="text-align: center; color: red; font-size: 16px;">
                <p>❌ ERROR</p>
                <p>{str(e)}</p>
            </div>
            """
            error_filepath = f'charts/error_map_{pd.Timestamp.now().strftime("%H%M%S")}.html'
            with open(error_filepath, 'w') as f:
                f.write(error_html)
            print(f"❌ Error generating map: {str(e)}")
            return error_filepath

    def get_training_data_from_geoglows(self, lat: float, lon: float) -> Optional[pd.DataFrame]:
        """
        Fetches GEOGloWS retrospective data to use as training data for local models.
        """
        print(f"Fetching historical simulation from GEOGloWS for Lat={lat}, Lon={lon}...")
        try:
            # Get River ID
            with requests.Session() as s:
                res = s.get('https://geoglows.ecmwf.int/api/v2/getriverid', params={'lat': lat, 'lon': lon})
                res.raise_for_status()
                river_id = res.json()['river_id']

            # Get retrospective data
            df_retro = geoglows.data.retrospective(river_id)
            df_retro.index = pd.to_datetime(df_retro.index)
            df_retro[df_retro < 0] = 0
            df_retro.rename(columns={'streamflow_m^3/s': 'streamflow_m3s'}, inplace=True)

            # We need covariates (precip, temp). Since GEOGloWS doesn't provide them,
            # we will use a weather API for historical weather data.
            # This makes the comparison fair as both models use external weather data.
            print("Fetching historical weather data to match retrospective streamflow...")
            weather_client = openmeteo_requests.Client()
            params = {
                "latitude": lat, "longitude": lon,
                "start_date": df_retro.index.min().strftime('%Y-%m-%d'),
                "end_date": df_retro.index.max().strftime('%Y-%m-%d'),
                "daily": ["precipitation_sum", "temperature_2m_mean"]
            }
            responses = weather_client.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
            response = responses[0]
            daily = response.Daily()

            df_weather = pd.DataFrame(data={
                "date": pd.to_datetime(daily.Time(), unit="s", utc=True).tz_convert(None),
                "precip_mm": daily.Variables(0).ValuesAsNumpy(),
                "temp_c": daily.Variables(1).ValuesAsNumpy(),
            }).set_index('date')

            # Combine streamflow and weather
            df_combined = df_retro.join(df_weather, how='inner')
            print(f"✅ Successfully created training dataset of shape {df_combined.shape}")
            return df_combined

        except Exception as e:
            print(f"❌ Failed to get training data: {e}")
            traceback.print_exc()
            return None

    def run_dl_ensemble_forecast(self, historical_df: pd.DataFrame, future_weather_df: pd.DataFrame,
                                 prediction_days: int) -> Optional[pd.DataFrame]:
        """
        Trains an ensemble of deep learning models and returns the forecast statistics.
        """
        print("\n--- Running Deep Learning Ensemble Forecast ---")
        models_to_run = {
            'N-BEATS': NBEATSModel(input_chunk_length=90, output_chunk_length=prediction_days, n_epochs=20,
                                   random_state=42),
            'Transformer': TransformerModel(input_chunk_length=90, output_chunk_length=prediction_days, n_epochs=20,
                                            random_state=42,
                                            d_model=16, nhead=4, num_encoder_layers=2, num_decoder_layers=2)
        }

        try:
            target_series = TimeSeries.from_series(historical_df['streamflow_m3s'], freq='D')
            covariate_series = TimeSeries.from_dataframe(
                pd.concat([historical_df[['precip_mm', 'temp_c']], future_weather_df]),
                value_cols=['precip_mm', 'temp_c'],
                freq='D'
            )

            # Normalize data
            scaler_target = Scaler()
            scaler_covariates = Scaler()
            target_series_scaled = scaler_target.fit_transform(target_series)
            covariate_series_scaled = scaler_covariates.fit_transform(covariate_series)

            all_predictions = []
            for name, model in models_to_run.items():
                print(f"Training {name} model...")
                model.fit(target_series_scaled, past_covariates=covariate_series_scaled, verbose=False)
                print(f"Predicting with {name} model...")
                prediction_scaled = model.predict(n=prediction_days, series=target_series_scaled,
                                                  past_covariates=covariate_series_scaled)
                prediction = scaler_target.inverse_transform(prediction_scaled)
                all_predictions.append(prediction.pd_series())

            # Combine predictions and calculate stats
            df_ensemble = pd.concat(all_predictions, axis=1)
            df_stats = pd.DataFrame(index=df_ensemble.index)
            df_stats['dl_median'] = df_ensemble.median(axis=1)
            df_stats['dl_q05'] = df_ensemble.quantile(0.05, axis=1)
            df_stats['dl_q95'] = df_ensemble.quantile(0.95, axis=1)

            print("✅ DL Ensemble forecast complete.")
            return df_stats

        except Exception as e:
            print(f"❌ Error during DL ensemble run: {e}")
            traceback.print_exc()
            return None

    def plot_forecast_comparison(self, historical_df: pd.DataFrame, geoglows_df: pd.DataFrame,
                                 dl_ensemble_df: pd.DataFrame) -> str:
        """
        Plots the GEOGloWS and DL ensemble forecasts on the same axes.
        """
        print("Generating forecast comparison plot...")
        with plt.style.context(PLOT_STYLE):
            fig, ax = plt.subplots(figsize=(15, 7))

            # Plot historical data
            ax.plot(historical_df.index[-30:], historical_df['streamflow_m3s'][-30:], 'k-',
                    label='Historical Simulation')

            # Plot GEOGloWS forecast
            ax.plot(geoglows_df.index, geoglows_df['stat_mean'], '--', color='red', label='GEOGloWS Mean')
            ax.fill_between(geoglows_df.index, geoglows_df['stat_min'], geoglows_df['stat_max'],
                            color='red', alpha=0.2, label='GEOGloWS Min-Max Range')

            # Plot DL Ensemble forecast
            ax.plot(dl_ensemble_df.index, dl_ensemble_df['dl_median'], '-', color=PRIMARY_COLOR,
                    label='DL Ensemble Median')
            ax.fill_between(dl_ensemble_df.index, dl_ensemble_df['dl_q05'], dl_ensemble_df['dl_q95'],
                            color=PRIMARY_COLOR, alpha=0.2, label='DL Ensemble 90% Interval')

            ax.set_title(f'Forecast Comparison: Deep Learning Ensemble vs. GEOGloWS')
            ax.set_ylabel('Streamflow (m³/s)')
            ax.set_xlabel('Date')
            ax.legend()
            ax.grid(True)

            filepath = f'charts/forecast_comparison_{self.station_id}.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Comparison plot saved to {filepath}")
            return filepath
    def _create_error_plot(self, message: str) -> str:
        with plt.style.context(PLOT_STYLE):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, f"❌ ERROR\n\n{message}", ha='center', va='center', color=SECONDARY_COLOR,
                    fontsize=12, wrap=True, bbox=dict(boxstyle='round,pad=0.5', fc='#f8d7da', ec=SECONDARY_COLOR))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            filepath = f'charts/error_{pd.Timestamp.now().strftime("%H%M%S")}.png'
            plt.savefig(filepath, bbox_inches='tight')
            plt.close(fig)
            return filepath