import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from openai import OpenAI
from shiny import App, ui, render, reactive
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import re

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load data once
def load_data():
    df = pd.read_excel(
        os.path.join("data", "HDD Raw Data.xlsx"),
        engine="openpyxl",
        sheet_name="background",
        usecols="A:H"
    )
    df.columns = df.columns.str.strip()
    df["Month End"] = pd.to_datetime(df["Month End"])
    df["Station Zip"] = df["Station Zip"].astype(str).str.zfill(5)  # Ensure 5-digit format
    return df

df = load_data()

# Extract unique ZIP codes from the dataset
def get_unique_zip_codes():
    # Get unique ZIP codes
    zip_codes = sorted(df["Station Zip"].unique())
    
    # Create a dictionary with ZIP codes as both keys and values
    # This will display the ZIP code to the user and use it as the value
    zip_dict = {code: code for code in zip_codes}
    
    return zip_dict

# Tool function to query HDD data
def get_hdd_by_zip(zipcode: str) -> str:
    # Ensure the zipcode is padded to match data format
    zipcode = zipcode.zfill(5)
    filtered = df[df["Station Zip"] == zipcode]
    if filtered.empty:
        return f"No data found for ZIP code {zipcode}."
    # Change from 12 months to 60 months (5 years)
    latest = filtered.sort_values("Month End").tail(60)[["Month End", "HDDs"]]
    return latest.to_string(index=False)

# Define the tool schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_hdd_by_zip",
            "description": "Fetch the last 12 months of Heating Degree Days (HDD) data for a ZIP code to analyze seasonal patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "zipcode": {
                        "type": "string",
                        "description": "The 5-digit ZIP code to retrieve HDD data for"
                    }
                },
                "required": ["zipcode"]
            }
        }
    }
]

# Get prediction for specified month
def call_openai(zipcode: str, target_date, adjust_for_warming=False):
    # Format the target date
    target_month_name = target_date.strftime("%B %Y")
    today = datetime.now()
    
    # Base prompt
    base_prompt = (
        f"Today is {today.strftime('%B %d, %Y')}. Please estimate the HDDs for {target_month_name} in ZIP code {zipcode} "
        f"using the historical data provided. Analyze seasonal patterns and trends in the data."
    )
    
    # Add global warming adjustment instruction if requested
    if adjust_for_warming:
        # Calculate years from now to target date
        years_forward = (target_date.year - today.year) + (target_date.month - today.month) / 12
        
        warming_prompt = (
            f"\n\nIMPORTANT: Adjust your estimate to account for global warming effects. "
            f"Assume a warming trend of approximately 1% fewer HDDs per year relative to historical averages. "
            f"Since you're predicting {years_forward:.1f} years into the future, "
            f"reduce your baseline estimate by approximately {years_forward:.1f}% to account for this warming trend."
        )
        base_prompt += warming_prompt
    
    # Add the structured output format instruction
    format_instruction = (
        f"\n\nAfter your analysis, end your response with a clear numerical prediction in this exact format:"
        f"\n\n{{PREDICTION: X}} where X is your single numeric estimate for {target_month_name}'s HDD value."
        f"\n\nFor example: {{PREDICTION: 123}} if you predict 123 HDDs."
    )
    
    user_prompt = base_prompt + format_instruction
    
    # Create system message with conditional instruction
    system_message = f"Today is {today.strftime('%B %d, %Y')}. You are helping estimate heating degree days for {target_month_name}."
    if adjust_for_warming:
        system_message += " Include global warming effects (1% reduction per year) in your estimate."
    system_message += " Always end your response with {PREDICTION: X} where X is your numerical estimate."
    
    # Step 1: Ask the model what tool it wants to use
    chat = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "get_hdd_by_zip"}}  # Force tool usage
    )

    first_response = chat.choices[0].message

    # Step 2: If the model wants to use the tool, call it
    if first_response.tool_calls:
        tool_call = first_response.tool_calls[0]
        
        try:
            args = json.loads(tool_call.function.arguments)
            zip_arg = args.get("zipcode")
            tool_output = get_hdd_by_zip(zip_arg)

            # Step 3: Send tool response back to OpenAI
            second_chat = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"Today is {today.strftime('%B %d, %Y')}. You are helping estimate heating degree days for {target_month_name}."},
                    {"role": "user", "content": user_prompt},
                    first_response,
                    {
                        "role": "tool", 
                        "tool_call_id": tool_call.id,
                        "name": "get_hdd_by_zip", 
                        "content": tool_output
                    }
                ]
            )

            response_content = second_chat.choices[0].message.content
            return response_content, target_date, tool_output
        except json.JSONDecodeError as e:
            return f"Error using tool: {e}", target_date, None
    else:
        return first_response.content, target_date, None

# Extract numeric value from OpenAI's response
def extract_hdd_value(response_text):
    """
    Extract HDD prediction value from model response with preference for structured format.
    """
    # Strategy 1: Look for the structured format we requested
    structured_pattern = r"\{PREDICTION:\s*(\d+)\}"
    structured_match = re.search(structured_pattern, response_text)
    if structured_match:
        value = int(structured_match.group(1))
        if 0 <= value <= 1500:  # Sanity check for reasonable range
            return value
    
    # Strategy 2: Fallback to looking for numbers in reasonable range
    numbers = re.findall(r'\b\d+\b', response_text)
    
    if numbers:
        for num in numbers:
            # Most likely the HDD value will be between 0 and 1500
            if 0 <= int(num) <= 1500:
                return int(num)
        
        # If we don't find a reasonable value, return the first number
        return int(numbers[0])
    
    return None  # No numeric value found

# Generate date options for month picker
def generate_month_options():
    today = datetime.now()
    options = {}
    
    # Generate 24 months into the future
    for i in range(1, 25):
        future_date = today + relativedelta(months=i)
        month_label = future_date.strftime("%B %Y")
        month_value = future_date.strftime("%Y-%m-01")  # First day of month
        options[month_value] = month_label
        
    return options

# Create Shiny app UI
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style(
            """
            .app-header {
                background-color: #2c3e50;
                color: white;
                padding: 15px 0;
                margin-bottom: 20px;
                text-align: center;
            }
            .prediction-box {
                margin-top: 20px;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f8f9fa;
            }
            .prediction-value {
                font-size: 24px;
                font-weight: bold;
                color: #e74c3c;
            }
            .form-group {
                margin-bottom: 20px;
            }
            .sidebar {
                background-color: #f8f9fa;
                padding: 15px;
                border-right: 1px solid #ddd;
            }
            .main-content {
                padding: 15px;
            }
            .main-content > .shiny-plot-output {
                margin-bottom: 30px;
            }
            """
        )
    ),
    ui.div(
        {"class": "app-header"},
        ui.h1("HDD Forecaster"),
        ui.p("Analyze historical Heating Degree Days and predict future values")
    ),
    
    # Using a simple row/column layout instead of sidebar
    ui.row(
        ui.column(3, 
            {"class": "sidebar"},
            ui.h3("Input Parameters"),
            ui.input_select("zipcode", "Select ZIP Code:", get_unique_zip_codes()),
            ui.input_select("target_month", "Select Month to Predict:", generate_month_options()),
            ui.input_checkbox("adjust_for_warming", "Adjust for global warming (1% per year)", value=False),
            ui.input_action_button("submit", "Generate Forecast", class_="btn-primary"),
            ui.hr(),
            ui.div(
                {"class": "prediction-box", "id": "prediction-container"},
                ui.h4("Prediction Results"),
                ui.output_text("prediction_month"),
                ui.div(
                    {"class": "prediction-value"},
                    ui.output_text("prediction_value")
                ),
                ui.output_text("raw_response")
            )
        ),
        ui.column(9, 
            {"class": "main-content"},
            ui.h3("Historical HDD Data with Prediction"),
            ui.output_plot("hdd_plot"),
            ui.div(
                {"style": "display: flex; justify-content: center; width: 100%;"},
                ui.div(
                    {"style": "width: 80%; max-width: 800px;"},
                    ui.h4("Historical Data (Last 36 Months)"),
                    ui.output_table("historical_data")
                )
            )
        )
    )
)

# Server logic
def server(input, output, session):
    # Store results in reactive values
    prediction_data = reactive.Value({
        "text": "",
        "value": None,
        "date": None,
        "raw_data": "",
        "zipcode": ""
    })
    
    @reactive.Effect
    @reactive.event(input.submit)
    def _():
        # Get the selected ZIP code
        zipcode = input.zipcode()
        if not zipcode:
            return
        
        # Get the selected month
        target_month_str = input.target_month()
        target_date = datetime.strptime(target_month_str, "%Y-%m-%d")
        
        # Get checkbox value
        adjust_for_warming = input.adjust_for_warming()
        
        # Get prediction from OpenAI with global warming adjustment if requested
        prediction_text, target_date, raw_data = call_openai(zipcode, target_date, adjust_for_warming)
        predicted_hdd = extract_hdd_value(prediction_text)
        
        # Update reactive values
        prediction_data.set({
            "text": prediction_text,
            "value": predicted_hdd,
            "date": target_date,
            "raw_data": raw_data if raw_data else "No data available",
            "zipcode": zipcode,
            "warming_adjusted": adjust_for_warming
        })
    
    @output
    @render.text
    def prediction_month():
        if prediction_data().get("date"):
            warming_text = " (warming-adjusted)" if prediction_data().get("warming_adjusted") else ""
            return f"Forecast for: {prediction_data()['date'].strftime('%B %Y')}{warming_text}"
        return "No prediction yet. Select a ZIP code and month, then click 'Generate Forecast'."
    
    @output
    @render.text
    def prediction_value():
        if prediction_data().get("value"):
            return f"{prediction_data()['value']} HDDs"
        return "---"
    
    @output
    @render.text
    def raw_response():
        if prediction_data().get("text"):
            return f"Model response: {prediction_data()['text']}"
        return ""
    
    @output
    @render.plot
    def hdd_plot():
        if not prediction_data().get("zipcode") or prediction_data().get("value") is None:
            # Return a clean empty plot (no placeholder text)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            return fig

        zipcode = prediction_data()["zipcode"]
        prediction_value = prediction_data()["value"]
        prediction_date = prediction_data()["date"]

        # Filter data for the specified ZIP code
        filtered_data = df[df["Station Zip"] == zipcode].sort_values("Month End")

        if filtered_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"No data found for ZIP code {zipcode}", 
                    ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            return fig

        # Get the last 60 months (5 years) of data for visualization
        recent_data = filtered_data.tail(60)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))  # Slightly wider to accommodate more data

        # Plot historical data
        ax.plot(recent_data["Month End"], recent_data["HDDs"], 'b-o', label='Historical HDDs', markersize=4)  # Smaller markers for clarity

        # Add prediction point
        ax.plot(prediction_date, prediction_value, 'r*', markersize=15, 
                label=f'Prediction for {prediction_date.strftime("%B %Y")}: {prediction_value} HDDs')

        # Customize the plot
        ax.set_title(f'HDD Data and Prediction for ZIP Code {zipcode} (5-Year History)')
        ax.set_xlabel('Month')
        ax.set_ylabel('Heating Degree Days (HDDs)')
        ax.grid(True, linestyle='--', alpha=0.7)

        # Format x-axis as dates - adjust to show fewer tick marks for readability
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Show tick every 6 months
        plt.xticks(rotation=45)

        # Add legend
        ax.legend()

        # Adjust layout
        plt.tight_layout()

        return fig

    
    @output
    @render.table
    def historical_data():
        if not prediction_data().get("zipcode"):
            return None
        
        zipcode = prediction_data()["zipcode"]
        filtered_data = df[df["Station Zip"] == zipcode].sort_values("Month End", ascending=False)
        
        if filtered_data.empty:
            return pd.DataFrame({"Message": ["No data found for this ZIP code"]})
        
        # Get available columns and ensure we only display ones that exist
        available_columns = ["Month End", "HDDs"]
        if "CDDs" in filtered_data.columns:
            available_columns.append("CDDs")
        
        # Show more months (36 instead of 12)
        display_data = filtered_data.head(36)[available_columns]
        
        # Format the Month End column
        display_data["Month End"] = display_data["Month End"].dt.strftime("%b %Y")
        
        # Replace NaN values with "Data Not Available"
        display_data = display_data.fillna("Data Not Available")
        
        return display_data

# Create and run the app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()