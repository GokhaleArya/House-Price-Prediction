import pandas as pd
import numpy as np
import pickle
from src.config import Trainer
from src.utils.logger import logging
from src.utils.exception import CustomException
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_model(model_path):
    '''
    Loads the model (dumped pickle).
    '''
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            return model
    except Exception as e:
        logging.info("Custom Exception in load_model.")
        raise CustomException(e, sys)
    
