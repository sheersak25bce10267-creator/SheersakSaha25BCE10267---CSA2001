# Intelligent Crop Recommendation System

## Overview
This project is a Machine Learning application designed to recommend the optimal crop for cultivation based on specific soil and environmental parameters. By analyzing metrics such as Nitrogen, Phosphorous, Potassium, pH, Temperature, Humidity, and Rainfall, the model removes human guesswork from agricultural planning and helps maximize yield.

## Project Structure
The project is built using standard Python scripts for a production-ready workflow and consists of the following files:

* `Crop_recommendation.csv`: The dataset containing 2,200 records of soil and weather metrics mapped to 22 unique crops.
* `analyze_data.py`: An Exploratory Data Analysis (EDA) script that generates visual insights, including feature correlation heatmaps and environmental distribution charts.
* `train_model.py`: The core machine learning pipeline. It preprocesses the data, performs hyperparameter tuning using Grid Search Cross-Validation, trains a Random Forest Classifier, and evaluates its accuracy.
* `crop_recommender_model.pkl`: The serialized machine learning model generated after training, ready for deployment.

## Setup Instructions
To run this project locally, ensure you have Python installed on your system. Open your terminal or command prompt, navigate to the project folder, and install the required dependencies:
  pandas
  numpy
  matplotlib
  seaborn
  scikit-learn
  jupyter
