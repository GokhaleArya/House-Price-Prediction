import pandas as pd
import numpy as np
import pickle
from src.config import Trainer
from src.utils.logger import logging
from src.utils.exception import CustomException
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(model_name, y_pred, y_true, n_features):
        try:
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            n = len(y_true)
            r2_adj = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

            print("\n")
            print("*"*50)
            print("-"*35)
            print(f'Model Name : {model_name}')
            print(f"-"*35)
            print(f"MAE : {mae}")
            print(f"MSE : {mse}")
            print(f"RMSE : {rmse}")
            print(f"R^2 : {r2}")
            print(f"Adjusted R^2 : {r2_adj}")
            
            return {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2,
                'Adjusted R²': r2_adj
            }
        except Exception as e:
              logging.info("Custom Exception in calculate_metrics")
              raise CustomException(e, sys)