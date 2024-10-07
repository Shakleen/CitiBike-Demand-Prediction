import os
from flask import Flask, render_template, request
import pandas as pd
import random

station_data = pd.read_csv(os.path.join("artifacts", "data", "Stations.csv"))
station_names = station_data.loc[:, "name"].tolist()
app = Flask(__name__)


@app.route("/")
def index():
    # Pass the list to the HTML template
    return render_template(
        "index.html",
        options=station_names,
        bike_demand=0,
        dock_demand=0,
    )


# Route to handle form submission when the "Check!" button is pressed
@app.route("/check", methods=["POST"])
def check_data():
    # Retrieve form data
    dropdown_option = request.form.get("dropdown_option")
    date = request.form.get("date")
    time = request.form.get("time")
    holiday = request.form.get("holiday")

    # Process the data (Here we just print to the console for testing)
    print(f"Dropdown option: {dropdown_option}")
    print(f"Date: {date}")
    print(f"Time: {time}")
    print(f"Holiday: {holiday}")

    # Replace to get demand predictions
    bike_demand = random.randint(1, len(station_names))
    dock_demand = random.randint(1, len(station_names))

    # Do something with the data and return a response
    return render_template(
        "index.html",
        options=station_names,
        bike_demand=bike_demand,
        dock_demand=dock_demand,
    )


if __name__ == "__main__":
    app.run(debug=True)
