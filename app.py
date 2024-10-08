import os
from flask import Flask, render_template, request
import pandas as pd
import pyspark
from delta import configure_spark_with_delta_pip
import pyspark.sql.functions as F

from src.prediction_pipeline.spark_predict_pipeline import predict
from src.data_pipeline.bronze_to_silver_transformer import create_time_features
from src.data_pipeline.silver_to_gold_transformer import cyclic_encode

builder = (
    pyspark.sql.SparkSession.builder.master("local[1]")
    .appName("flask_app")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )
    .config("spark.driver.memory", "15g")
    .config("spark.sql.shuffle.partitions", "6")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()

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
    station_name = request.form.get("dropdown_option")
    date = request.form.get("date")
    time = request.form.get("time")
    holiday = request.form.get("holiday")

    # Process the data (Here we just print to the console for testing)
    print(f"Dropdown option: {station_name}")
    print(f"Date: {date}")
    print(f"Time: {time}")
    print(f"Holiday: {holiday}")
    print(station_data.loc[station_data.name == station_name, "latitude"].tolist()[0])

    bike_demand, dock_demand = get_demands(station_name, date, time, holiday)

    # Do something with the data and return a response
    return render_template(
        "index.html",
        options=station_names,
        bike_demand=bike_demand,
        dock_demand=dock_demand,
    )


def get_demands(station_name, date, time, holiday):
    df = spark.createDataFrame(
        data=[
            [
                station_data.loc[
                    station_data.name == station_name, "latitude"
                ].tolist()[0],
                station_data.loc[
                    station_data.name == station_name, "longitude"
                ].tolist()[0],
                f"{date} {time}",
                1 if holiday is not None else 0,
            ]
        ],
        schema="latitude float, longitude float, time string, is_holiday integer",
    )
    df = df.withColumn("time", F.to_timestamp(F.col("time"), "yyyy-MM-dd HH:mm"))
    df = create_time_features(df)

    # Replace to get demand predictions
    bike_demand, dock_demand = predict(
        df,
        "random_forest",
        os.path.join("artifacts", "pipelines", "gold_pipeline"),
    )
    return bike_demand, dock_demand


if __name__ == "__main__":
    app.run(debug=True)
