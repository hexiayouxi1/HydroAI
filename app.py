import numpy as np
import gradio as gr
import os
import data_agent
import analysis_toolkit
import llm_dispatcher
import warnings
import pandas as pd
import re
import inspect

warnings.filterwarnings("ignore", category=UserWarning)

# --- Global State & Helpers ---
TOOLKIT_INSTANCE, WEATHER_PROVIDER_INSTANCE, FUTURE_WEATHER_DATA = None, None, None
BASE_MODEL_PATH = r"C:/Users/Administrator/.cache/modelscope/hub/models/Qwen/Qwen2-7B-Instruct"
ADAPTER_PATH = "./qwen-ts-analyst-lora"

def create_sample_data(filepath='sample_data.csv'):
    """
    Creates a more realistic sample dataset for comprehensive testing.
    This new version includes:
    - A long-term linear trend in streamflow.
    - More realistic, "spiky" precipitation using a Gamma distribution.
    - A 'tracer_concentration' column that co-varies with precipitation events.
    """
    if os.path.exists(filepath):
        print(f"'{filepath}' already exists. Using existing file.")
        return filepath

    print("Creating enhanced sample dataset for testing...")
    dates = pd.date_range(start='2018-01-01', end='2022-12-31', freq='D')
    num_days = len(dates)

    # 1. Streamflow: Seasonal pattern + long-term trend + noise
    seasonal_flow = 40 * np.sin(np.linspace(0, 10 * np.pi, num_days))  # 5 years = 10 half-cycles
    long_term_trend = -np.linspace(0, 15, num_days)  # Simulate a slight decline over 5 years
    noise_flow = np.random.normal(0, 5, num_days)
    streamflow = 50 + seasonal_flow + long_term_trend + noise_flow
    streamflow[streamflow < 0] = 0  # Flow cannot be negative

    # 2. Precipitation: Spiky and more realistic
    precip = np.random.gamma(0.4, 3.0, num_days)
    precip[np.random.rand(num_days) < 0.85] = 0  # 85% of days are dry
    # Add a couple of guaranteed large storm events for testing separation
    storm_indices = [np.where(dates == pd.to_datetime('2021-08-15'))[0][0],
                     np.where(dates == pd.to_datetime('2022-04-20'))[0][0]]
    for idx in storm_indices:
        precip[idx:idx + 3] = [30, 45, 20] + np.random.rand(3) * 5

    # 3. Temperature: Seasonal pattern
    temp = 15 + 10 * np.sin(np.linspace(0, 10 * np.pi - np.pi / 2, num_days)) + np.random.normal(0, 1.5, num_days)

    # 4. Tracer Concentration: Simulates dilution during rain events
    C_OLD = 100  # Baseflow (old water) concentration
    C_NEW = 10  # Precipitation (new water) concentration
    precip_influence = pd.Series(precip).rolling(window=3, min_periods=1).mean() / 25  # Recent rain effect (scaled)
    precip_influence = precip_influence.clip(0, 1)  # Bounded between 0 and 1
    tracer_concentration = C_OLD * (1 - precip_influence) + C_NEW * precip_influence

    data = {
        'date': dates,
        'streamflow_m3s': streamflow,
        'precip_mm': precip,
        'temp_c': temp,
        'tracer_concentration': tracer_concentration
    }

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print("✅ Enhanced sample dataset created.")
    return filepath

# --- Backend Functions ---
def initialize_system(filepath, date_col, flow_col, precip_col, temp_col, station_id, lat, lon):
    global TOOLKIT_INSTANCE, WEATHER_PROVIDER_INSTANCE
    try:
        if not os.path.exists('charts'): os.makedirs('charts')
        prepared_data = data_agent.load_and_prepare_data(filepath, date_col, flow_col, precip_col, temp_col)
        TOOLKIT_INSTANCE = analysis_toolkit.RunoffToolkit(prepared_data, station_id)
        WEATHER_PROVIDER_INSTANCE = data_agent.WeatherProvider(lat, lon)
        return "✅ System Initialized. Ready for analysis."
    except Exception as e:
        return f"❌ Initialization Failed: {e}"

def fetch_weather_data(prediction_days):
    global FUTURE_WEATHER_DATA
    if WEATHER_PROVIDER_INSTANCE is None: return "Provider not initialized.", None
    FUTURE_WEATHER_DATA = WEATHER_PROVIDER_INSTANCE.get_future_forecast(prediction_days)
    if FUTURE_WEATHER_DATA is not None: return f"✅ Fetched {len(FUTURE_WEATHER_DATA)} days of weather data.", FUTURE_WEATHER_DATA
    return "❌ Failed to fetch weather data.", None

def train_model(model_name: str):
    """Triggers model training and returns status, metrics, and a plot."""
    if TOOLKIT_INSTANCE is None:
        return "Toolkit not initialized.", None, None
    status, metrics, plot = TOOLKIT_INSTANCE.train_forecasting_model(model_name=model_name)
    return status, metrics, plot


def chat_interface(message, history):
    history.append([message, None])
    if TOOLKIT_INSTANCE is None and "get_external_station_data" not in message:  # Allow external data fetching before init
        history[-1][1] = "❌ Error: System not initialized. Please load data in 'Part 1'."
        return history, None, None, None, None

    intent = llm_dispatcher.get_intent(message)
    function_name = intent.get("function")
    args = intent.get("args", {})

    response_text, plot_path1, plot_path2, map_html, df_result = "Sorry, I couldn't understand that.", None, None, None, None

    try:
        if function_name == "get_geoglows_analysis":
            lat = args.get("lat")
            lon = args.get("lon")
            analysis_type = args.get("analysis_type")
            if not all([lat, lon, analysis_type]):
                response_text = "❌ Error: For GEOGloWS analysis, I need a latitude, longitude, and analysis type (e.g., retrospective, forecast_stats)."
            else:
                # The function returns an HTML string of the plot
                html_plot = data_agent.run_geoglows_analysis(lat, lon, analysis_type)
                if "<p style='color:red;'>" in html_plot: # Check for error message
                    response_text = "An error occurred while fetching GEOGloWS data."
                else:
                    response_text = f"✅ Here is the GEOGloWS {analysis_type.replace('_', ' ')} plot."
                map_html = html_plot # Display the plot in the HTML component

        # --- NEW: Handle External Data Fetching ---
        elif function_name == "compare_forecasts_with_geoglows":
            lat = args.get("lat")
            lon = args.get("lon")
            forecast_days = args.get("forecast_days", 7)  # Default to 7 days

            if not all([lat, lon]):
                response_text = "❌ Error: I need a latitude and longitude to run the forecast comparison."
            else:
                # This workflow is complex, so we'll provide step-by-step feedback
                response_text = "Starting forecast comparison workflow...\n"

                # 1. Get GEOGloWS forecast
                geoglows_html = data_agent.run_geoglows_analysis(lat, lon, 'forecast_stats')
                # We need the data, not the plot. Let's add a way to get the data directly.
                # (This requires a small modification to data_sources.py)
                geoglows_df = data_agent.get_geoglows_data_only(lat, lon, 'forecast_stats')

                if geoglows_df is None or geoglows_df.empty:
                    response_text += "❌ Failed to get GEOGloWS forecast data. Aborting."
                    history[-1][1] = response_text
                    return history, None, None, None, None
                response_text += "✅ Step 1/4: Fetched GEOGloWS forecast data.\n"

                # 2. Get training data
                training_df = TOOLKIT_INSTANCE.get_training_data_from_geoglows(lat, lon)
                if training_df is None or training_df.empty:
                    response_text += "❌ Failed to get historical data for training. Aborting."
                    history[-1][1] = response_text
                    return history, None, None, None, None
                response_text += "✅ Step 2/4: Fetched historical training data.\n"

                # 3. Get future weather
                weather_provider = data_agent.WeatherProvider(lat, lon)
                future_weather = weather_provider.get_future_forecast(forecast_days)
                if future_weather is None or future_weather.empty:
                    response_text += "❌ Failed to get future weather data. Aborting."
                    history[-1][1] = response_text
                    return history, None, None, None, None
                response_text += "✅ Step 3/4: Fetched future weather forecast.\n"

                # 4. Run DL ensemble and plot
                dl_ensemble_df = TOOLKIT_INSTANCE.run_dl_ensemble_forecast(training_df, future_weather, forecast_days)
                if dl_ensemble_df is None or dl_ensemble_df.empty:
                    response_text += "❌ Failed to run the Deep Learning ensemble."
                else:
                    plot_path1 = TOOLKIT_INSTANCE.plot_forecast_comparison(training_df, geoglows_df, dl_ensemble_df)
                    response_text += "✅ Step 4/4: DL Ensemble complete. Comparison plot is ready."
        elif function_name == "get_external_station_data":
            source = args.get("source")
            identifier = args.get("identifier")
            if not source or not identifier:
                response_text = "❌ Error: The 'source' and 'identifier' are required to fetch external data."
            else:
                primary_df, secondary_df = data_agent.fetch_external_data(source, identifier)
                if primary_df is not None and not primary_df.empty:
                    df_result = primary_df
                    response_text = f"✅ Successfully fetched data from {source}."
                    if secondary_df is not None and not secondary_df.empty:
                        response_text += " Found historical and forecast data. Displaying primary dataset below."
                else:
                    response_text = f"❌ Failed to retrieve data from {source} for identifier '{identifier}'."

        elif function_name == "predict_with_model":
            if FUTURE_WEATHER_DATA is None:
                response_text = "⚠️ Weather data not fetched. Please fetch in 'Part 1'."
            else:
                days = args.get("prediction_days", 7)
                plot_path1 = TOOLKIT_INSTANCE.predict_with_model(prediction_days=days,
                                                                future_weather_df=FUTURE_WEATHER_DATA)
                if TOOLKIT_INSTANCE.trained_model_name:
                    response_text = f"Generated forecast for {days} days using {TOOLKIT_INSTANCE.trained_model_name} model."
                else:
                    response_text = f"Generated forecast for {days} days, but no model has been trained yet."

        elif function_name == "train_forecasting_model":
            model_name_arg = args.get("model_name", "TFT")
            status, _, _ = TOOLKIT_INSTANCE.train_forecasting_model(model_name=model_name_arg)
            response_text = status

        elif hasattr(TOOLKIT_INSTANCE, function_name):
            target_function = getattr(TOOLKIT_INSTANCE, function_name)
            func_params = inspect.signature(target_function).parameters
            filtered_args = {key: value for key, value in args.items() if key in func_params}
            print(f"Calling function '{function_name}' with filtered arguments: {filtered_args}")
            result = target_function(**filtered_args)

            if isinstance(result, str):
                if result.endswith('.png'):
                    plot_path1 = result
                    response_text = "Here is the requested plot."
                elif result.endswith('.html'):
                    with open(result, 'r') as f:
                        map_html = f.read()
                    response_text = "Here is the interactive map."
                else:
                    response_text = result
            elif isinstance(result, tuple) and all(isinstance(p, str) and p.endswith('.png') for p in result):
                plot_path1 = result[0]
                plot_path2 = result[1] if len(result) > 1 else None
                response_text = f"Generated {len(result)} SHAP plots. Displaying summary and dependence plots."
            elif isinstance(result, pd.DataFrame):
                df_result = result
                response_text = "Here is the statistical analysis."
            elif isinstance(result, dict):
                if "plot_path" in result:
                    plot_path1 = result.pop("plot_path")
                df_result = pd.DataFrame([result])
                response_text = "Here is the analysis."
            else:
                response_text = str(result)

        else:
            response_text = f"I don't know how to perform the action: '{function_name}'. Please try rephrasing."

    except Exception as e:
        import traceback
        response_text = f"An error occurred while executing '{function_name}': {e}\n{traceback.format_exc()}"

    history[-1][1] = response_text
    return history, plot_path1, plot_path2, map_html, df_result

# --- Custom CSS ---
CUSTOM_CSS = """
/* Change the main background */
body, .gradio-container { background-color: #E9ECEF; }

/* Style the tabs container */
.tab-nav { 
    background-color: #343A40;
    border-radius: 8px 8px 0 0;
    padding: 5px;
}

/* Style individual tab buttons */
.tab-nav button {
    background-color: #6C757D;
    color: white !important;
    border: 2px solid #495057 !important;
    border-radius: 6px !important;
    margin: 0 5px !important;
    transition: all 0.3s ease;
}

/* Style the selected tab button */
.tab-nav button.selected {
    background-color: #007BFF !important;
    border-color: #0056b3 !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* Style the tab content area */
.tabitem {
    background-color: white;
    border: 1px solid #DEE2E6;
    border-top: none;
    padding: 20px;
    border-radius: 0 0 8px 8px;
}
"""

# --- Gradio UI ---
with gr.Blocks(theme='soft', title="HydroAI", css=CUSTOM_CSS) as demo:
    gr.Markdown("# 🌊 HydroAI: A Hydrological Intelligence System")

    with gr.Tabs():
        with gr.TabItem("Part 1: Data & Setup", elem_id="tab1"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 1. Load Custom Historical Data")
                    file_upload = gr.File(label="Upload .csv File", value=create_sample_data())
                    station_id_input = gr.Textbox(label="Station Identifier", value="SampleStation")
                    lat_input = gr.Number(label="Latitude", value=40.71)
                    lon_input = gr.Number(label="Longitude", value=-74.01)
                    with gr.Accordion("Advanced Column Mapping", open=False):
                        date_col = gr.Textbox(label="Date Column", value='date')
                        flow_col = gr.Textbox(label="Flow Column", value='streamflow_m3s')
                        precip_col = gr.Textbox(label="Precipitation Column", value='precip_mm')
                        temp_col = gr.Textbox(label="Temperature Column", value='temp_c')
                    init_button = gr.Button("Initialize System", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown("### 2. Fetch Future Data (for Prediction)")
                    prediction_days_slider = gr.Slider(1, 30, value=7, step=1, label="Days to Fetch")
                    fetch_button = gr.Button("Fetch Weather Data")
                    fetch_status = gr.Textbox(label="Fetch Status", interactive=False)
                    weather_df_output = gr.DataFrame(label="Fetched Weather Data")
            init_status = gr.Textbox(label="System Status", interactive=False)

        with gr.TabItem("Part 2: Model Training & Evaluation", elem_id="tab2"):
            gr.Markdown("## Train and Evaluate a Forecasting Model")
            gr.Markdown("Select a model, click 'Train', and review its performance on the validation set below.")
            with gr.Row():
                with gr.Column(scale=1):
                    model_name_dropdown = gr.Dropdown(
                        label="Select Forecasting Model",
                        choices=['TFT', 'NBEATS', 'BLOCKRNN', 'TCN', 'TRANSFORMER', 'LIGHTGBM', 'VARIMA'],
                        value='TFT',
                        info="Choose a model. All use weather data."
                    )
                    train_button = gr.Button("Train & Evaluate Model", variant="primary")
                    train_status = gr.Textbox(label="Training Status", interactive=False, lines=4)
                with gr.Column(scale=2):
                    gr.Markdown("### Evaluation Results")
                    evaluation_plot = gr.Image(label="Model Performance on Validation Set", type="filepath")
                    metrics_df_output = gr.DataFrame(label="Performance Metrics")

        with gr.TabItem("Part 3: Analysis & Prediction Q&A", elem_id="tab3"):
            gr.Markdown("## Chat with the HydroAI Assistant")
            gr.Markdown("Examples: `explain the model`, `get water storage for Nile Basin`, `get data for USGS site 09380000`, `fetch geoglows forecast for 15.6, 32.558`，`show precipitation map for Nile Basin`")
            chatbot = gr.Chatbot(height=400)
            with gr.Row():
                plot_output1 = gr.Image(label="Plot Output 1", type="filepath")
                plot_output2 = gr.Image(label="Plot Output 2 (for SHAP)", type="filepath")
                map_output = gr.HTML(label="Map Output")
            df_output = gr.DataFrame(label="Data Output")
            msg = gr.Textbox(label="Your Request")
            msg.submit(
                chat_interface,
                [msg, chatbot],
                [chatbot, plot_output1, plot_output2, map_output, df_output]
            )

    init_inputs = [file_upload, date_col, flow_col, precip_col, temp_col, station_id_input, lat_input, lon_input]
    init_button.click(initialize_system, inputs=init_inputs, outputs=init_status)
    fetch_button.click(fetch_weather_data, inputs=[prediction_days_slider], outputs=[fetch_status, weather_df_output])
    train_button.click(
        train_model,
        inputs=[model_name_dropdown],
        outputs=[train_status, metrics_df_output, evaluation_plot]
    )
    demo.load(lambda: llm_dispatcher.load_llm_model(BASE_MODEL_PATH, ADAPTER_PATH), outputs=init_status)

if __name__ == "__main__":

    demo.launch(debug=True,share=True)
