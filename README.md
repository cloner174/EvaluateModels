# EvaluateModels: Time Series Analysis and Visualization

## Introduction

This Python script, created with respect to the divine, offers comprehensive tools for evaluating and visualizing the performance of various forecasting models. Aimed at data scientists and analysts, it provides functionalities to plot mean absolute error (MAE) over different forecast horizons, compare models, visualize trend and seasonality components, and much more. This document serves as a guide for using the `EvaluateModels` class within your projects.

## Features
- **Plot MAE Over Time**: Visualizes the MAE for different forecast horizons to assess model accuracy.
- **Plot Confidence Intervals**: Displays confidence intervals for forecasts, with an option to include actual values for comparison.
- **Compare Models**: Compares the MAE of different models across forecast horizons.
- **Trend and Seasonality Visualization**: Plots the trend, seasonal, and observed components of time series data.
- **Feature Importance**: Visualizes the importance of features used by the model.
- **Anomaly Detection**: Identifies and visualizes anomalies where the absolute error exceeds a defined threshold.

## Prerequisites

Before using the `EvaluateModels` class, ensure that you have installed the following Python packages:

- `matplotlib`
- `numpy`

## Usage

To utilize the functionalities provided by the `EvaluateModels` class, first import the necessary packages and then instantiate the class:

    import matplotlib.pyplot as plt
    import numpy as np
    from EvaluateModels import EvaluateModels
    evaluator = EvaluateModels()

# After instantiation 

You can call any of the methods provided by the class as needed. For example, to plot the MAE over different forecast horizons:

    mae_dict = {1: 0.1, 2: 0.15, 10: 0.2}
    evaluator.plot_mae_over_time(mae_dict)

# Contact Information

For further inquiries or contributions, feel free to reach out through the following channels:

Telegram:     https://t.me/PythonLearn0
Email:        cloner174.org@gmail.com

# Acknowledgments

This script is developed aiming to contribute positively to the learning community's efforts in data analysis and model evaluation.
