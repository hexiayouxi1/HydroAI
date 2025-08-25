# hydro_ai_project/llm_dispatcher.py
from datetime import datetime, timedelta # Add timedelta to imports

import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

MODEL, TOKENIZER = None, None


def load_llm_model(base_model_path, adapter_path):
    """Loads the quantized base model and merges the LoRA adapter."""
    global MODEL, TOKENIZER
    if MODEL is not None: return "Model already loaded."
    print("Loading LLM model...")
    try:
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_compute_dtype=torch.bfloat16)
        MODEL = AutoModelForCausalLM.from_pretrained(base_model_path, quantization_config=quant_config,
                                                     trust_remote_code=True, device_map="auto")
        TOKENIZER = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        MODEL = PeftModel.from_pretrained(MODEL, adapter_path)
        return "✅ LLM Model loaded successfully."
    except Exception as e:
        return f"❌ Error loading LLM model: {e}"


# ### FIX: Update the system prompt to include date arguments ###
SYSTEM_PROMPT = """You are an intelligent dispatcher for a hydrological analysis toolkit. Your task is to translate a user's request into a single, structured JSON object representing a function call.
Your entire output MUST be a single, valid JSON object and nothing else.

Available functions:
- `get_statistics(start_date: str, end_date: str)`: For statistical summaries.
- `plot_hydrograph(start_date: str, end_date: str)`: For flow visualization.
- `calculate_flood_frequency(return_period: int)`: For flood return periods.
- `detect_trend(data_column: str, period: str)`: To find trends.
- `separate_hydrograph_components(event_date: str, c_new: float, c_old: float)`: For hydrograph separation.
- `predict_with_model(prediction_days: int)`: For all forecasting tasks.
- `train_forecasting_model(model_name: str)`: To train a model.
- `explain_forecast()`: To 'explain', 'interpret', or 'show drivers of' the model.
- `get_satellite_precipitation(region: str, start_date: str, end_date: str)`: For satellite rain data.
- `calculate_regional_et(region: str)`: For regional evapotranspiration.
- `get_water_storage_anomaly(region: str)`: For GRACE-like water storage data.
- `map_surface_water_dynamics(water_body_name: str)`: For lake/reservoir surface area.
- `display_map(data_type: str, region: str, start_date: str, end_date: str)`: To display maps.
- `get_external_station_data(source: str, identifier: str)`: **Fetches real-time data from specific national services.**
  - `source` MUST be one of: 'USGS', 'CANADA', 'AUSTRALIA', 'UK', 'INDIA'. **DO NOT USE 'GEOGLOWS' here.**
  - `identifier` is the station ID.
- `get_geoglows_analysis(lat: float, lon: float, analysis_type: str)`: **Use this for ALL GEOGloWS requests. It fetches data and creates a plot.**
  - `analysis_type` must be one of: 'retrospective', 'forecast_stats', 'forecast_ensembles', 'forecast'.
  - Extract latitude and longitude from the user query.
- `compare_forecasts_with_geoglows(lat: float, lon: float, forecast_days: int)`: **Use this to run a local deep learning (DL) ensemble and compare its forecast against the GEOGloWS forecast for a specific location.**
Example 1: user says "get data for USGS site 09380000"
{"function": "get_external_station_data", "args": {"source": "USGS", "identifier": "09380000"}}
Example 2: user says "show the geoglows retrospective for 15.6, 32.558"
{"function": "get_geoglows_analysis", "args": {"lat": 15.6, "lon": 32.558, "analysis_type": "retrospective"}}
Example 3: user says "what are the geoglows forecast ensembles at lat 40.7 lon -74.1"
{"function": "get_geoglows_analysis", "args": {"lat": 40.7, "lon": -74.1, "analysis_type": "forecast_ensembles"}}
Example 4: user says "fetch the forecast stats from geoglows for 15.6, 32.558"
{"function": "get_geoglows_analysis", "args": {"lat": 15.6, "lon": 32.558, "analysis_type": "forecast_stats"}}
Example 5: user says "what's the latest from the Canadian station 05BB001?"
{"function": "get_external_station_data", "args": {"source": "CANADA", "identifier": "05BB001"}}
Example 6: user says "get indian CWC data"
{"function": "get_external_station_data", "args": {"source": "INDIA", "identifier": "all"}}
Example 7: user says "compare a 10-day DL forecast against geoglows for 45.5, -122.6"
{"function": "compare_forecasts_with_geoglows", "args": {"lat": 45.5, "lon": -122.6, "forecast_days": 10}}
"""


# ### FIX: Add a new helper function for date extraction ###
def extract_dates_from_query(query: str) -> dict:
    """Extracts start and end dates from natural language."""
    args = {}
    query = query.lower()

    # 1. Look for specific YYYY-MM-DD dates
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    dates_found = re.findall(date_pattern, query)
    if len(dates_found) == 1:
        args['start_date'] = dates_found[0]
    elif len(dates_found) >= 2:
        args['start_date'] = dates_found[0]
        args['end_date'] = dates_found[1]
        return args  # Prioritize specific dates

    # 2. Look for relative terms like "last X years/months"
    year_match = re.search(r'last\s+(\d+)\s+year', query)
    if year_match:
        years = int(year_match.group(1))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        args['start_date'] = start_date.strftime('%Y-%m-%d')
        args['end_date'] = end_date.strftime('%Y-%m-%d')
        return args

    month_match = re.search(r'last\s+(\d+)\s+month', query)
    if month_match:
        months = int(month_match.group(1))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)
        args['start_date'] = start_date.strftime('%Y-%m-%d')
        args['end_date'] = end_date.strftime('%Y-%m-%d')
        return args

    return args


def get_intent(query: str):
    """Uses the LLM to get the function call information from a user query."""
    if MODEL is None or TOKENIZER is None:
        return {"function": "error", "args": {"message": "LLM not loaded."}}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]
    text = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = TOKENIZER([text], return_tensors="pt").to(MODEL.device)

    outputs = MODEL.generate(inputs.input_ids, max_new_tokens=100, pad_token_id=TOKENIZER.eos_token_id)
    response_text = TOKENIZER.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            intent_data = json.loads(json_match.group(0))
            # ### FIX: Post-process to add extracted dates ###
            # This makes the system more robust. Even if the LLM forgets the date args,
            # our regex logic will add them.
            date_args = extract_dates_from_query(query)
            if 'args' not in intent_data:
                intent_data['args'] = {}
            intent_data['args'].update(date_args)  # Add/overwrite with regex-extracted dates
            return intent_data
        else:
            return {"function": "unknown", "args": {"reason": "Could not parse LLM response."}}
    except json.JSONDecodeError:
        return {"function": "unknown", "args": {"reason": f"Invalid JSON from LLM: {response_text}"}}