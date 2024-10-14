# Extracts zip file and saves it to 'extracts'
# Train test splits and saves in new folder 'clean'
import zipfile
from src.utils.logger import logging
from src.utils.exception import CustomException
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def ZipFileExtractor(zip_path, extract_path):
    '''
    Extracts zip file to specified location.
    Inputs:
    zip_path : The path where zip is located.
    extract_path : The path where zip is to be extracted.
    '''
    try :
        with zipfile.ZipFile(zip_path, 'r') as files:
            files.extractall(extract_path)
            print("Zip files extracted!")
    except Exception as e:
        logging.info("Custom Exception in ZipFileExtractor.")
        raise CustomException(e, sys)
    
def RawDataSplitter(raw_data_path, target_col, test_data_path='', test=False):
    '''
    Returns data shuffled, splitted in form 
    X_train, X_test, y_train, y_test
    '''
    try:
        data = pd.read_csv(raw_data_path).set_index('Id')
        X_train, X_test, y_train, y_test = train_test_split(data.drop(target_col, axis=1), 
                                                                data[target_col],
                                                                test_size=0.2,
                                                                random_state=1234, shuffle=True)
        if test==False:
            return X_train, y_train, X_test, y_test
        
        if test==True:
            test_data = pd.read_csv(test_data_path).set_index('Id')
            X_test = test_data
            return X_train, X_test, y_train
    except Exception as e:
        logging.info("Custom Exception in RawDataSplitter.")
        raise CustomException(e, sys)
