from catboost import CatBoostRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.linear_model import LinearRegression


def get_model(name, params):

    if name=='CatBoost':
        default_params = {
            'loss_function':'RMSE',
            'thread_count':4,
            'early_stopping_rounds': 10, 
            'verbose': False
        }
        model = CatBoostRegressor(**{**default_params,**params})
    elif name=='LGBM':
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
        }
        model = LGBMRegressor(**{**default_params,**params})
    

    return model

