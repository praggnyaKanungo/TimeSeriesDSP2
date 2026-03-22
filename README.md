# DS4002 Project 2- Time Series

Goal: We aim to understand the relationship between federal government spending and various health and econimic outcomes. We collect datasets on life expectancy, death rate, and poverty rate across several decades, split by certain demographic factors such as race, sex, and age. We use federal spending data on different categories of economic/healthcare spending to try and predict the evolution of these health and economic factors. 

## Software and Platform ##

The software for this repository is primarily written in Jupyter notebooks and python scripts. Additionally, we utilize the python packages Numpy, Pandas, Matplotlib, and Scipy. All code outputs were produced by running python notebooks or python files locally on a Mac environment.

## A Map of Documentation

The documentation of this project is broken up in to 3 primary folders. The DATA folder contains all of the data in both its original and cleaned form. The combined dataset, including all variables used in analysis is in DATA/Combined_Dataset.csv. Python scripts are stored in the SCRIPTS folder, these scripts are used for both exploratory data analysis, as well as the full modeling and analysis of the dataset. The SCRIPTS folder contains scripts which are used for data cleaning, producing EDA plots, as well as modeling using linear regression and ARIMA. Finally, the outputs of the project are stored in the OUTPUTS folder. These outputs include plots produced during EDA to understand the dataset, as well as outputs from the regression and ARIMA modeling.

## Instructions for Reproducing Results

To reproduce these results, the python scripts and notebooks in the SCRIPTS folder must be run. To produce the initial EDA plots, the EDA_Plots_Spending.ipynb notebook can be run. In order to perform the analysis, one first must run the scripts pertaining to data cleaning, and then the scripts pertaining to analysis. The data cleaning scripts are, in order, reformat_health_data.py, formatted_final_dataset.py, and lag_feature_dataset.py. The analysis scripts are, in order, linear_regression.py, and ARIMA_predictions.py. The scripts correctly reference the DATA folder to source data correctly, and running these scripts produces outputs in the OUTPUT folder, reproducing our analysis


