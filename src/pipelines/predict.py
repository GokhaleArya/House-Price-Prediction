import pandas as pd
import numpy as np
import pickle
from src.config import Trainer
from src.utils.logger import logging
from src.utils.exception import CustomException
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def predict(model, X_train, y_train, X_test, y_test):
    '''
    Prints summary and returns predictions.
    '''
    try:
        model = model
        print('Predicting......')
        predictions = model.predict(X_test)
        print('Done Predicting!')
        
        return predictions
    
    except Exception as e:
        logging.info('Custom Exception in predict.')
        raise CustomException(e, sys)