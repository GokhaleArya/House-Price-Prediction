# Processes the data. Returns and stores cleaned data to 
# The techniques used were decided in '/notebooks'
import pandas as pd
import numpy as np
from src.utils.logger import logging
from src.utils.exception import CustomException
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

def Imputer(obj_cols, num_cont_cols, num_dis_cols, X_train, y_train, X_test, y_test):
    '''
    Imputs obj_cols with None, numerical_continous_cols with mode and 
    numerical_discrete_cols with median.
    Expects obj_cols, num_cont_cols, num_dis_cols, X_train, y_train, X_test, y_test for train
    Put y_test = None for prediction/testing
    '''
    try:
        obj_imputer = SimpleImputer(strategy='constant', fill_value='None')
        num_cont_imputer = SimpleImputer(strategy='most_frequent')
        num_dis_imputer = SimpleImputer(strategy='median')

        if (y_test) is not None:
            X_train[obj_cols] = obj_imputer.fit_transform(X_train[obj_cols])
            X_test[obj_cols] = obj_imputer.transform(X_test[obj_cols])

            X_train[num_cont_cols] = num_cont_imputer.fit_transform(X_train[num_cont_cols])
            X_test[num_cont_cols] = num_cont_imputer.transform(X_test[num_cont_cols])

            X_train[num_dis_cols] = num_dis_imputer.fit_transform(X_train[num_dis_cols])
            X_test[num_dis_cols] = num_dis_imputer.transform(X_test[num_dis_cols])

            return X_train, y_train, X_test, y_test
        
        if (y_test) is None:
            X_train[obj_cols] = obj_imputer.fit_transform(X_train[obj_cols])
            X_test[obj_cols] = obj_imputer.transform(X_test[obj_cols])

            X_train[num_cont_cols] = obj_imputer.fit_transform(X_train[obj_cols])
            X_test[num_cont_cols] = obj_imputer.transform(X_test[obj_cols])

            X_train[num_dis_cols] = obj_imputer.fit_transform(X_train[obj_cols])
            X_test[num_dis_cols] = num_dis_imputer.transform(X_test[num_dis_cols])
            
            return X_train, y_train, X_test
    except Exception as e:
        logging.info('Custom Exception in Imputer.')
        raise CustomException(e, sys)
    

def LabelEncode(obj_cols, X_train, y_train, X_test, y_test):
    '''
    Label Encodes the columns in obj_cols.
    Expects obj_cols, X_train, y_train, X_test, y_test for training.
    Put y_test=None for testing/prediction.
    '''
    try:
        label_encoders = {}
        for col in obj_cols:
            label_encoder = LabelEncoder()
            X_train[col] = label_encoder.fit_transform(X_train[col])
            label_encoders[col] = label_encoder
            X_test[col] = X_test[col].apply(lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1 if pd.notna(x) else -1)
        if y_test is not None:
            return X_train, y_train, X_test, y_test
        if y_test is None:
            return X_train, y_train, X_test
    except Exception as e:
        logging.info("Custom Exception in LabelEncode.")
        raise CustomException(e, sys)
    
def StandardScale(numerical_continous_cols, X_train, y_train, X_test, y_test):
    '''
    Scales the input to mean=0 and S.D.=1
    Expects numerical_continous_cols, X_train, y_train, X_test, y_test for training
    And y_test=None for test/prediction.
    '''
    try:
        scaler = StandardScaler()
        X_train[numerical_continous_cols] = scaler.fit_transform(X_train[numerical_continous_cols])
        X_test[numerical_continous_cols] = scaler.transform(X_test[numerical_continous_cols])

        if y_test is not None:
            return X_train, y_train, X_test, y_test
        if y_test is None:
            return X_train, y_train, X_test
    except Exception as e:
        logging.info("Custom Exception in StandardScale.")
        raise CustomException(e, sys)
    
def PrintHead(data):
    '''
    Prints top 5 rows of the data passed.
    '''
    try:
        print(data.head())
    except Exception as e:
        logging.info('Custom Exception in PrintHead.')
        raise CustomException(e, sys)











