# MMA3001 - Building Occupancy Forecasting

Author: NS

Unit: MMA3001 - Numerical Methods and Machine Learning

## About the Project

For this project, I used building occupancy sensor data to predict how many people would be in a space one hour later.

I used Python to process the data, prepare the information needed for predictions, and compare three different forecasting methods. The main goal was to see whether machine learning could predict occupancy more accurately than simpler methods.

## Dataset

The dataset provided for the assignment contains approximately 9.09 million sensor records from November 2023 to April 2026.

The sensor readings were not recorded at regular hourly intervals, so I converted them into hourly occupancy values.

For each hour, the program uses the most recent sensor reading if it is no more than two hours old. If there is no recent reading, that hour is treated as missing instead of assuming the space is empty.

The original dataset is not included in this repository.

## Method

The program first reads the dataset from a ZIP file and removes invalid records.

It then creates hourly occupancy values and uses previous measurements to prepare the data for forecasting.

The information used for prediction includes:

- Current occupancy
- Occupancy one and two hours earlier
- Occupancy 24 hours earlier
- Hour of the day
- Day of the week
- Whether it is a weekend

The aim is to predict occupancy one hour ahead.

I compared three forecasting methods.

1. Persistence method - assumes the occupancy will remain the same as the current hour.

2. Previous-day method - uses the occupancy from the same hour on the previous day.

3. Machine learning method - uses a histogram-based gradient boosting model from scikit-learn to learn patterns from the data.

The data was divided in time order, with 60% for training, 20% for validation and 20% for testing.

## Results

After processing the dataset, 50,398 usable prediction examples were obtained across four building spaces.

The test results were:

Persistence method:
MAE = 0.7465
RMSE = 1.7242

Previous-day method:
MAE = 0.8714
RMSE = 1.9223

Machine learning method:
MAE = 0.7280
RMSE = 1.3785

MAE and RMSE measure how far the predictions are from the actual occupancy values. Lower values indicate better predictions.

The machine learning method performed best overall. Compared with the persistence method, it reduced MAE by approximately 2.5% and RMSE by approximately 20%.

The comparison graphs are saved in the results folder.

## Project Files

src/pipeline.py - Main Python program that processes the data, trains the models and generates the results.

tests/test_pipeline.py - Contains the automated tests.

results/ - Contains the prediction results, evaluation metrics and graphs.

docs/ - Contains the generated documentation for the Python functions.

requirements.txt - Lists the Python packages needed to run the program.

LICENSE - Project licence.

## Running the Program

First, install the required Python packages by entering the following command in the terminal:

pip install -r requirements.txt

Then run the program using the original occupancy dataset ZIP file:

python -m src.pipeline --zip "path/to/occupancy.zip"

Replace the example path with the actual location of the dataset.

The program will save the results and graphs in the results folder.

To run the automated tests, enter:

python -m pytest tests/

## Limitations

There are some limitations to the project.

The sensor data contains missing readings, and the amount of available data differs between building spaces.

Only four spaces had enough usable data for the forecasting experiment.

The timestamps were processed in UTC rather than local building time, which may affect the daily occupancy patterns.

The results may also differ when using data from other buildings or future time periods.

## AI Assistance

I used ChatGPT during this project for assistance with writing python code and debugging, developing the forecasting methods, creating tests, and preparing documentation. I also used it to increase my understanding of the errors, results and overall implementation of the project.

The code was executed and checked using automated tests.
