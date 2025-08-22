# HydroAI
a specialized conversational AI framework for hydrological data analysis

Readme.md
Step 1: Setup
1.	Make sure all your files (app.py, analysis_toolkit.py, data_agent.py, llm_dispatcher.py) are in the same directory.
2.	If it exists, delete the old sample_data.csv so the new one is generated.
3.	Install all required libraries: pip install gradio "darts[pytorch,lgbm]" pandas numpy pymannkendall shap openmeteo-requests requests-cache retry-requests "transformers[bitsandbytes]" peft accelerate
4.	Run the application from your terminal: python app.py
5.	Open the provided URL (e.g., http://127.0.0.1:7860) in your web browser.
Step 2: Walkthrough and Function Testing
Part 1: Data & Setup Tab
1.	Initialize the System: The sample_data.csv file should already be selected. Click the "Initialize System" button. You should see "✅ System Initialized. Ready for analysis." in the status box.
2.	Fetch Weather Data: Click the "Fetch Weather Data" button. The status should update to "✅ Fetched 7 days of weather data." and a table of future weather will appear.
Part 2: Model Training & Evaluation Tab
1.	Train a Model: Select LIGHTGBM from the dropdown menu (this is important for testing the explain_forecast function later).
2.	Click "Train & Evaluate Model".
3.	Observe the results:
o	The "Training Status" box will show "✅ LIGHTGBM trained successfully...".
o	The "Performance Metrics" table will populate with R², NSE, MAE, and RMSE values.
o	The "Model Performance on Validation Set" plot will appear, showing how well the model's forecast (red dashed line) matches the actual validation data (green line).
Part 3: Analysis & Prediction Q&A Tab
Now, use the chatbot to test each major function of the RunoffToolkit. Type the following prompts into the message box and press Enter.
1.	Test get_statistics:
o	Prompt: show me statistics for the last 3 years
o	Expected Output: A data table will appear in the "Data Output" box with stats like Mean Flow, Max Flow, etc.
2.	Test plot_hydrograph:
o	Prompt: plot the hydrograph for 2021
o	Expected Output: A plot of the daily streamflow for the year 2021 will appear in the "Plot Output" box.
3.	Test calculate_flood_frequency:
o	Prompt: what is the 100-year flood?
o	Expected Output: A data table with the estimated discharge for a 100-year return period.
4.	Test detect_trend:
o	Prompt: is there a significant annual trend in streamflow
o	Expected Output: A data table showing the trend analysis results (e.g., trend: 'decreasing', is_significant: True, p_value, etc.). This should work because we added a trend to the synthetic data.
5.	Test separate_hydrograph_components:
o	Prompt: separate the hydrograph for the storm on 2021-08-15 using c_new=10 and c_old=100
o	Expected Output: A stacked area plot showing the "New Water" and "Old Water" contributions to a flood event, along with a data table summarizing the volumes. This tests the new tracer_concentration column.
6.	Test predict_with_model:
o	Prompt: forecast 7 days
o	Expected Output: A plot showing the historical runoff, the 7-day predicted runoff, and the future precipitation forecast.
7.	Test explain_forecast (SHAP):
o	Prompt: explain the forecast model
o	Expected Output: Since this function returns two plots, the Gradio interface will likely only show the first one (the summary plot). The agent's logic would need to be updated to handle multiple plot outputs. However, checking your charts directory, you should find both shap_summary...png and shap_dependence...png. This confirms the function ran successfully.
8.	Test Remote Sensing Functions:
o	Prompt: show me the water storage anomaly for the Nile Basin
o	Expected Output: A GRACE plot showing a declining trend in water storage.
o	Prompt: map the surface water dynamics of Lake Powell
o	Expected Output: A bar chart showing the annual change in the lake's surface area.
o	Prompt: calculate regional ET for the Amazon
o	Expected Output: A plot showing the seasonal cycle of evapotranspiration.
