#      #               #                        In the name of God                    #      #
#
#Author: cloner174 (Github.com/cloner174)
#Contact: cloner174.org@gmail.com
#
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Any


class EvaluateModels:
    
    def plot_mae_over_time(self, mae_dict: Dict[int, float]) -> None:
        """
        Plot the Mean Absolute Error (MAE) for different forecast horizons.
        
        Parameters:
            mae_dict (Dict[int, float]): Dictionary where keys are forecast horizons
                                         (e.g., 1, 2, 10 days) and values are the
                                         corresponding MAEs.
        
        Raises:
            ValueError: If `mae_dict` is empty.
        """
        if not mae_dict:
            raise ValueError("MAE dictionary cannot be empty.")
        
        # Sorting by forecast horizon for a clean plot
        horizons = sorted(mae_dict.keys())
        maes = [mae_dict[h] for h in horizons]
        
        plt.figure(figsize=(10, 6))
        plt.plot(horizons, maes, marker='o', linestyle='-', color='b')
        plt.title('MAE Over Different Forecast Horizons')
        plt.xlabel('Forecast Horizon (days)')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.xticks(horizons)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_confidence_intervals(
        self, 
        forecast_horizon: int, 
        lower_bounds: List[float], 
        upper_bounds: List[float], 
        actuals: Optional[List[float]] = None
    ) -> None:
        """
        Plot confidence intervals for a specific forecast horizon, optionally comparing to actual values.
        
        Parameters:
            forecast_horizon (int): The forecast horizon (e.g., 30 days).
            lower_bounds (List[float]): Lower bounds of the confidence interval.
            upper_bounds (List[float]): Upper bounds of the confidence interval.
            actuals (Optional[List[float]]): Actual values for comparison (optional).
        
        Raises:
            ValueError: If provided lists do not have the same length.
        """
        if len(lower_bounds) != len(upper_bounds) or (actuals is not None and len(actuals) != len(lower_bounds)):
            raise ValueError("All provided lists must have the same length.")
        
        x_vals = list(range(len(lower_bounds)))
        
        plt.figure(figsize=(10, 6))
        plt.fill_between(x_vals, lower_bounds, upper_bounds, color='skyblue', alpha=0.4, label='Confidence Interval')
        if actuals is not None:
            plt.plot(x_vals, actuals, color='orange', label='Actual Values')
        plt.title(f'Confidence Intervals for {forecast_horizon}-day Forecast')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_model_comparisons(self, model_maes: Dict[str, Dict[int, float]]) -> None:
        """
        Compare the MAE of different models across forecast horizons.
        
        Parameters:
            model_maes (Dict[str, Dict[int, float]]): Dictionary where keys are model names and values
                                                      are dictionaries mapping forecast horizons to MAEs.
        
        Raises:
            ValueError: If `model_maes` is empty.
        """
        if not model_maes:
            raise ValueError("Model MAEs dictionary cannot be empty.")
        
        plt.figure(figsize=(10, 6))
        for model, mae_dict in model_maes.items():
            # Sort horizons to ensure consistent plotting order
            horizons = sorted(mae_dict.keys())
            maes = [mae_dict[h] for h in horizons]
            plt.plot(horizons, maes, marker='o', linestyle='-', label=model)
        
        plt.title('Model Comparative MAE Over Different Forecast Horizons')
        plt.xlabel('Forecast Horizon (days)')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_trend_seasonality(
        self, 
        trend: List[float], 
        seasonal: List[float], 
        observed: List[float]
    ) -> None:
        """
        Plot the trend, seasonal, and observed components of the time series.
        
        Parameters:
            trend (List[float]): Trend component of the time series.
            seasonal (List[float]): Seasonal component of the time series.
            observed (List[float]): Observed values of the time series.
        
        Raises:
            ValueError: If the lengths of the provided lists are not equal.
        """
        if not (len(trend) == len(seasonal) == len(observed)):
            raise ValueError("Trend, seasonal, and observed lists must have the same length.")
        
        plt.figure(figsize=(14, 8))
        
        plt.subplot(311)
        plt.plot(observed, label='Observed', color='blue')
        plt.title('Observed Data')
        plt.legend(loc='best')
        plt.grid(True)
        
        plt.subplot(312)
        plt.plot(trend, label='Trend', color='green')
        plt.title('Trend Component')
        plt.legend(loc='best')
        plt.grid(True)
        
        plt.subplot(313)
        plt.plot(seasonal, label='Seasonality', color='red')
        plt.title('Seasonal Component')
        plt.legend(loc='best')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, model: Any, feature_names: List[str]) -> None:
        """
        Plot the feature importances of a trained model.
        
        Parameters:
            model (Any): Trained model that must have a `feature_importances_` attribute.
            feature_names (List[str]): Names corresponding to the features.
        
        Raises:
            AttributeError: If `model` does not have a `feature_importances_` attribute.
        """
        if not hasattr(model, 'feature_importances_'):
            raise AttributeError("The model does not have a 'feature_importances_' attribute.")
        
        importances = model.feature_importances_
        indices = np.argsort(importances)
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.show()

    def detect_anomalies(
        self, 
        predictions: List[float], 
        actuals: List[float], 
        threshold: float
    ) -> List[int]:
        """
        Detect and visualize anomalies where the absolute error exceeds a threshold.
        
        Parameters:
            predictions (List[float]): Predicted values.
            actuals (List[float]): Actual observed values.
            threshold (float): Threshold for absolute error to consider as an anomaly.
        
        Returns:
            List[int]: Indices where anomalies were detected.
        
        Raises:
            ValueError: If `predictions` and `actuals` have different lengths.
        """
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals lists must have the same length.")
        
        # Calculate absolute errors and detect indices exceeding the threshold
        errors = np.abs(np.array(predictions) - np.array(actuals))
        anomalies = np.where(errors > threshold)[0]
        
        plt.figure(figsize=(10, 6))
        plt.plot(actuals, label='Actual Values', color='blue')
        plt.scatter(anomalies, np.array(actuals)[anomalies], color='red', label='Anomalies')
        plt.title('Anomaly Detection in Predictions')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return anomalies.tolist()

#end#
