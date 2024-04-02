#                            #                   #                                   In the name of God    #      #
#
#Github.com/cloner174
#cloner174.org@gmail.com
#
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional

class EvaluateModels:
    
    def plot_mae_over_time(self, mae_dict: Dict[int, float]) -> None:
        """
        Plot the Mean Absolute Error (MAE) for different forecast horizons.
        
        Parameters:
        - mae_dict: A dictionary where keys are forecast horizons (e.g., 1, 2, 10 days) as integers,
          and values are the corresponding MAEs as floats.
        
        Raises:
        - ValueError: If `mae_dict` is empty.
        """
        if not mae_dict:
            raise ValueError("MAE dictionary cannot be empty.")
        
        horizons = list(mae_dict.keys())
        maes = list(mae_dict.values())
        
        plt.figure(figsize=(10, 6))
        plt.plot(horizons, maes, marker='o', linestyle='-', color='b')
        plt.title('MAE Over Different Forecast Horizons')
        plt.xlabel('Forecast Horizon (days)')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.xticks(horizons)
        plt.grid(True)
        plt.show()
    
    
    def plot_confidence_intervals(self, forecast_horizon: int, lower_bounds: List[float], upper_bounds: List[float], actuals: Optional[List[float]] = None) -> None:
        """
        Plot confidence intervals for a specific forecast horizon, optionally including actual values for comparison.
        
        Parameters:
        - forecast_horizon: The specific forecast horizon (e.g., 30 days) as an integer.
        - lower_bounds: Lower bounds of the confidence interval as a list of floats.
        - upper_bounds: Upper bounds of the confidence interval as a list of floats.
        - actuals: Actual values for the forecast horizon as a list of floats, optional.
        
        Raises:
        - ValueError: If `lower_bounds` and `upper_bounds` (and optionally `actuals`, if provided) do not have the same length.
        """
        if len(lower_bounds) != len(upper_bounds) or (actuals is not None and len(actuals) != len(lower_bounds)):
            raise ValueError("All provided lists must have the same length.")
        
        plt.figure(figsize=(10, 6))
        plt.fill_between(range(len(lower_bounds)), lower_bounds, upper_bounds, color='skyblue', alpha=0.4, label='Confidence Interval')
        if actuals is not None:
            plt.plot(actuals, color='orange', label='Actual Values')
        plt.title(f'Confidence Intervals for {forecast_horizon}-day Forecast')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    
    
    def plot_model_comparisons(self, model_maes: Dict[str, Dict[int, float]]) -> None:
        """
        Compare the MAE of different models across forecast horizons.
        
        Parameters:
        - model_maes: A dictionary where keys are model names (as strings) and values are dictionaries
          of forecast horizons (as integers) to MAEs (as floats).
        
        Raises:
        - ValueError: If `model_maes` is empty.
        """
        if not model_maes:
            raise ValueError("Model MAEs dictionary cannot be empty.")
        
        plt.figure(figsize=(10, 6))
        for model, mae_dict in model_maes.items():
            horizons = list(mae_dict.keys())
            maes = list(mae_dict.values())
            plt.plot(horizons, maes, marker='o', linestyle='-', label=model)
        
        plt.title('Model Comparative MAE Over Different Forecast Horizons')
        plt.xlabel('Forecast Horizon (days)')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    
    def plot_trend_seasonality(self, trend: List[float], seasonal: List[float], observed: List[float]) -> None:
        """
        Plot the trend, seasonal, and observed components of the time series.
        
        Parameters:
        - trend: Trend component of the time series as a list of floats.
        - seasonal: Seasonal component of the time series as a list of floats.
        - observed: Observed values of the time series as a list of floats.
        
        Raises:
        - ValueError: If `trend`, `seasonal`, and `observed` do not have the same length.
        """
        if not (len(trend) == len(seasonal) == len(observed)):
            raise ValueError("Trend, seasonal, and observed lists must have the same length.")
        
        plt.figure(figsize=(14, 8))
        plt.subplot(311)
        plt.plot(observed, label='Observed')
        plt.legend(loc='best')
        plt.subplot(312)
        plt.plot(trend, label='Trend')
        plt.legend(loc='best')
        plt.subplot(313)
        plt.plot(seasonal, label='Seasonality')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
    
    
    def plot_feature_importance(self, model, feature_names: List[str]) -> None:
        """
        Plot the feature importances of the model.
        
        Parameters:
        - model: The trained model with a feature_importances_ attribute.
        - feature_names: List of names corresponding to the features used by the model.
        
        Raises:
        - AttributeError: If `model` does not have a `feature_importances_` attribute.
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
        plt.show()
    
    
    def detect_anomalies(self, predictions: List[float], actuals: List[float], threshold: float) -> List[int]:
        """
        Detect anomalies where the absolute error exceeds the threshold and visualize them.
        
        Parameters:
        - predictions: Predicted values as a list of floats.
        - actuals: Actual values as a list of floats.
        - threshold: Error threshold for detecting anomalies as a float.
        
        Returns:
        - List of indices representing the positions of anomalies in the input lists.
        
        Raises:
        - ValueError: If `predictions` and `actuals` do not have the same length.
        """
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals lists must have the same length.")
        
        errors = np.abs(np.array(predictions) - np.array(actuals))
        anomalies = np.where(errors > threshold)[0]
        
        plt.figure(figsize=(10, 6))
        plt.plot(actuals, label='Actual Values')
        plt.scatter(anomalies, np.array(actuals)[anomalies], color='r', label='Anomalies')
        plt.title('Anomaly Detection in Predictions')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        
        return anomalies.tolist()

#end#