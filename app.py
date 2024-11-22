from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for rendering plots
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the ARIMA model
with open("arima_model.pkl", "rb") as f:
    model = pickle.load(f)

# Route for the home page
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle forecasts
@app.route("/forecast", methods=["POST"])
def forecast():
    # Get user input (e.g., number of months to forecast)
    months = int(request.form.get("months", 12))

    # Generate forecast
    forecast = model.get_forecast(steps=months)
    forecast_values = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()

    # Generate future dates
    last_date = pd.Timestamp("1960-12-01")  # Replace with the actual last date of your dataset
    forecast_index = pd.date_range(last_date, periods=months + 1, freq="MS")[1:]

    # Plot the forecast using the Agg backend
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_index, forecast_values, label="Forecast", color="orange")
    plt.fill_between(
        forecast_index,
        forecast_conf_int.iloc[:, 0],
        forecast_conf_int.iloc[:, 1],
        color="orange",
        alpha=0.3,
    )
    plt.title("ARIMA Forecast")
    plt.xlabel("Date")
    plt.ylabel("Predicted Values")
    plt.legend()

    # Save plot to a string buffer
    img = BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template("forecast.html", plot_url=plot_url, months=months)

if __name__ == "__main__":
    app.run(debug=True)
