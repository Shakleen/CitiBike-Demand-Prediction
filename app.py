import os
from flask import Flask, render_template
import pandas as pd

station_data = pd.read_csv(os.path.join("artifacts", "data", "Stations.csv"))
app = Flask(__name__)

@app.route("/")
def index():
    # Python list to populate the dropdown
    options_list = station_data.loc[:, "name"].tolist()

    # Pass the list to the HTML template
    return render_template("index.html", options=options_list)


if __name__ == "__main__":
    app.run(debug=True)
