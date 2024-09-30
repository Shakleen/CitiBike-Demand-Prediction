# CitiBike-Demand-Prediction
An End-to-End Machine Learning project where I predict demand of bikes at citibike stations at hourly level.

## Business Objective

The success of the bike-sharing business hinges on ensuring that users can readily access bikes when needed and find available docking stations at the end of their rides. Consequently, there is a dual demand for both bike availability and empty docking stations. This project aims to forecast these demands on an hourly basis for each station in New York City. Accurate demand predictions will enhance the coordination of bike redistribution efforts, ultimately leading to increased profitability.

## Technical Details

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-Cluster%20Computing-orange)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-Data%20Lake-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking%20Experiment-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey)
![PyTest](https://img.shields.io/badge/PyTest-Testing-green)

### Technologies Used
1. **PySpark** and **DeltaLake** to manage data ingestion and transformation
1. **MLFlow** for model development
1. **Flask** to create front end
1. **Git** and **GitHub** for project management
1. **GitHub Actions** for CI/CD
1. **PyTest** for writing tests

### Medallion Data Architecture

![Medallion Data Architecture](Diagrams/Medallion_Architecture.jpeg)

## Acknowlegements
1. [Unit testing PySpark code using Pytest](https://engineeringfordatascience.com/posts/pyspark_unit_testing_with_pytest/)