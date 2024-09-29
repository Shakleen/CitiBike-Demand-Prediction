# CitiBike-Demand-Prediction
An End-to-End Machine Learning project where I predict demand of bikes at citibike stations at hourly level.

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-Cluster%20Computing-orange)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-Data%20Lake-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking%20Experiment-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey)
![PyTest](https://img.shields.io/badge/PyTest-Testing-green)



## Objective
Accurately predict the demand of bikes on 
1. A specific date and time of day and
1. A specific bike station

Here, the demand is defined as the following
1. **Bike Demand**: Number of bikes that will be taken from this station to go to another station
1. **Docking Demand**: Number of bikes that will come to this station to dock.

## Technologies Used
1. **PySpark** and **DeltaLake** to manage data ingestion and transformation
1. **MLFlow** for model development
1. **Flask** to create front end
1. **Git** and **GitHub** for project management
1. **GitHub Actions** for CI/CD
1. **PyTest** for writing tests