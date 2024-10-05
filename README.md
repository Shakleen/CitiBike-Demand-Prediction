# CitiBike-Demand-Prediction

Citi Bike, New York City’s bike share system, provides a network of docking stations across the city, allowing users to unlock bikes from one station and return them to any other. The success of the bike-sharing business hinges on ensuring that users can readily access bikes when needed and find available docking stations at the end of their rides. Consequently, there is a dual demand for both bike availability and empty docking stations. 

**This project aims to forecast these demands on an hourly basis for each station in New York City. Accurate demand predictions will enhance the coordination of bike redistribution efforts, ultimately leading to increased profitability.**

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-Cluster%20Computing-orange)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-Data%20Lake-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking%20Experiment-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey)
![PyTest](https://img.shields.io/badge/PyTest-Testing-green)
![Azure](https://img.shields.io/badge/Azure-Cloud_Service-blue)
![Docker](https://img.shields.io/badge/Docker-Containerization-blue)

## Table of Contents

1. [Overview](#overview)
1. [Technologies Used](#technologies-used)
1. [Data](#data)
1. [Technical Details](#technical-details)
1. [Acknoledgements](#acknowlegements)

## Overview

The system allows a user to query demand statistics for a particular station for a particular hour of any day. The interaction between different parts of the system is as follows:

![Sequence Diagram](Diagrams/Sequence_Diagram.jpeg)

## Data

The dataset contains **200 million rows** of structured data accessed through citibike data api. The data spans from 2013 to September 2024. There are mainly 4 types of data contained with in the structured dataset which are as follow:
1. **Time**: When the ride started and ended.
1. **Station**: Which startion the ride started and ended. This includes the name of station and it's geographical location.
1. **Membership**: Whether the rider was a member or not

## Technical Details

### Data Pipeline

The following diagrams shows the data processing pipeline used in this project. In brief here's what I have done for each step
1. **Raw to bronze**: Enforce schema and select useful columns.
1. **Bronze to Silver**: Split station into its own table. 
1. **Silver to gold**: Create useful features. Aggregate to create demand numbers.

Gold data is used to train machine learning models for predictive task.

![Medallion Data Architecture](Diagrams/Medallion_Architecture.jpeg)

## Technologies Used
1. **PySpark** and **DeltaLake** to manage data ingestion and transformation
1. **MLLib** and **MLFlow** for model development, tracking, and evluation
1. **Flask** to create front end
1. **GitHub Actions** for CI/CD
1. **PyTest** for writing tests
1. **Docker** to create containerized images
1. **Azure** for deployment

## Acknowlegements
1. [Unit testing PySpark code using Pytest](https://engineeringfordatascience.com/posts/pyspark_unit_testing_with_pytest/)