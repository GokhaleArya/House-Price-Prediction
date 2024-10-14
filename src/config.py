class ZipExtraction:
    zip_path = './zips/house-prices-advanced-regression-techniques.zip'
    extract_path = './extracts/raw'

class RawTrainData:
    raw_path = './extracts/raw/train.csv'
    target_column = 'SalePrice'

class Preprocessor:
    numerical_continous_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                                '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
                                'OpenPorchSF', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

    target_column =['SalePrice']

    numerical_discrete_cols = ['MSSubClass','OverallQual','OverallCond','YearBuilt','YearRemodAdd',
                               'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd',
                               'Fireplaces','GarageYrBlt','GarageCars','EnclosedPorch','MoSold','YrSold']

    obj_cols = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
                 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 
                 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
                 'SaleType', 'SaleCondition']
    
class Trainer:
    baseline_xgb_path = './src/models/XGBoost_baseline.pkl'
    baseline_lgb_path = './src/models/LightGBM_baseline.pkl'
    baseline_cat_path = './src/models/CatBoost_baseline.pkl'
    baseline_rf_path = './src/models/Random_Forest_baseline.pkl'
