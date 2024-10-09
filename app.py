import os
from flask import Flask, render_template, request
import pandas as pd
import pyspark
from delta import configure_spark_with_delta_pip
import pyspark.sql.functions as F
from datetime import datetime
import plotly.graph_objs as go
import plotly.io as pio

from src.prediction_pipeline.spark_predict_pipeline import predict
from src.data_pipeline.bronze_to_silver_transformer import create_time_features

builder = (
    pyspark.sql.SparkSession.builder
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
    return render_template("index.html", options=station_names, plot_html=None)


@app.route("/check", methods=["POST"])
def check_data():
    station_name = request.form.get("dropdown_option")
    holiday = request.form.get("holiday")

    demand_df = get_demands(station_name, holiday)
    plot_html = get_demand_plot(demand_df, station_name)

    return render_template("index.html", options=station_names, plot_html=plot_html)


def get_demand_plot(demand_df, station_name):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=demand_df["time"],
            y=demand_df["predicted_bike_demand"],
            mode='lines',
            name="Bike Demand",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=demand_df["time"],
            y=demand_df["predicted_dock_demand"],
            mode='lines',
            name="Dock Demand",
        )
    )

    fig.update_layout(
        title=f"Bike and Dock Demand at {station_name}<br><sup>Over the next 24 hours</sup>",
        xaxis_title="Time",
        yaxis_title="Demand",
        template="plotly_white",
        title_x=0.5,
        legend=dict(orientation="h", yanchor="top", xanchor="center", y=-0.2, x=0.5),
        barmode="group",
    )

    return pio.to_html(fig, full_html=False)


def get_demands(station_name, holiday):
    now = datetime.now()
    time_range = [
        (now + pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M") for i in range(24)
    ]
    df = get_dataframe(station_name, holiday)
    df = df.withColumn("time", F.to_timestamp(F.col("time"), "yyyy-MM-dd HH:mm"))
    df = create_time_features(df)

    # Replace to get demand predictions
    bike_demand, dock_demand = predict(
        df,
        "random_forest",
        os.path.join("artifacts", "pipelines", "gold_pipeline"),
    )

    demand_df = unify_demand_dataframe(time_range, bike_demand, dock_demand)

    return demand_df


def unify_demand_dataframe(time_range, bike_demand, dock_demand):
    bike_demand["time"] = time_range
    dock_demand["time"] = time_range

    demand_df = pd.merge(left=bike_demand, right=dock_demand, on="time", how="inner")
    demand_df["predicted_bike_demand"] = demand_df["predicted_bike_demand"].map(round)
    demand_df["predicted_dock_demand"] = demand_df["predicted_dock_demand"].map(round)
    return demand_df


def get_dataframe(station_name, holiday):
    latitude = station_data.loc[station_data.name == station_name, "latitude"].tolist()[
        0
    ]
    longitude = station_data.loc[
        station_data.name == station_name, "longitude"
    ].tolist()[0]
    now = datetime.now()

    df = spark.createDataFrame(
        data=[
            [
                latitude,
                longitude,
                (now + pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M"),
                1 if holiday is not None else 0,
            ]
            for i in range(24)
        ],
        schema="latitude float, longitude float, time string, is_holiday integer",
    )

    return df


if __name__ == "__main__":
    app.run(debug=True)
