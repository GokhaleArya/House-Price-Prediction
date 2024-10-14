import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.data.data_loader import ZipFileExtractor, RawDataSplitter
from src.data.data_process import Imputer, LabelEncode, StandardScale, PrintHead
from src.config import ZipExtraction, RawTrainData, Preprocessor, Trainer
from src.pipelines.train import load_model
from src.pipelines.predict import predict
from src.pipelines.evaluate import calculate_metrics
import warnings
warnings.filterwarnings('ignore')


if __name__=="__main__":
    try:
        logging.info('The execution has started')

        ZipFileExtractor(ZipExtraction.zip_path, ZipExtraction.extract_path)
        logging.info('Zip Data Extracted.')

        X_train, y_train, X_test, y_test = RawDataSplitter(RawTrainData.raw_path, RawTrainData.target_column)
        logging.info('Data split done and ready for processing.')

        PrintHead(X_train)
        
        X_train, y_train, X_test, y_test = Imputer(Preprocessor.obj_cols, Preprocessor.numerical_continous_cols,
                                                   Preprocessor.numerical_discrete_cols,
                                                   X_train, y_train, X_test, y_test)
        logging.info("Data Imputing done.")

        X_train, y_train, X_test, y_test = LabelEncode(Preprocessor.obj_cols, X_train, y_train,
                                                       X_test, y_test)
        logging.info("Label Encoding done.")

        X_train, y_train, X_test, y_test = StandardScale(Preprocessor.numerical_continous_cols,
                                                        X_train, y_train, X_test, y_test)
        PrintHead(X_train)
        
        xgb_baseline = load_model(Trainer.baseline_xgb_path)
        cat_baseline = load_model(Trainer.baseline_cat_path)
        lgb_baseline = load_model(Trainer.baseline_lgb_path)
        rf_baseline = load_model(Trainer.baseline_rf_path)
        logging.info("All models loaded.")

        xgb_baseline_preds = predict(xgb_baseline, X_train, y_train, X_test, y_test)
        cat_baseline_preds = predict(cat_baseline, X_train, y_train, X_test, y_test)
        lgb_baseline_preds = predict(lgb_baseline, X_train, y_train, X_test, y_test)
        rf_baseline_preds = predict(rf_baseline, X_train, y_train, X_test, y_test)
        logging.info("All predictions made.")

        calculate_metrics("XGBoost-Baseline", xgb_baseline_preds, y_test, len(X_train.columns))
        calculate_metrics("CatBoost-Baseline", cat_baseline_preds, y_test, len(X_train.columns))
        calculate_metrics("LightGBM-Baseline", lgb_baseline_preds, y_test, len(X_train.columns))
        calculate_metrics("RandomForest-Baseline", rf_baseline_preds, y_test, len(X_train.columns))
        logging.info("Metrics displayed")

        logging.info("Done for now.")

    except Exception as e:
        logging.info("Custom Exception in app.py")
        raise CustomException(e, sys)