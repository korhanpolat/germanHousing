# from skopt import BayesSearchCV
from skopt.callbacks import DeltaYStopper, CheckpointSaver
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize, gbrt_minimize
from skopt import load, dump
import os
import numpy as np
from sklearn.model_selection import cross_val_score
import time
import pandas as pd
from train_utils import validate_model
from model_config import get_model


SPACES = {
    'LGBM' : [
        Real(0.001, 0.05, 'log-uniform', name='learning_rate'), 
        Integer(1, 20, name='max_depth'),
        Integer(100, 1100, name= 'n_estimators'),                   # Number of boosted trees to fit
        Integer(2, 120, name='num_leaves'),    
        # Integer(10, 1000, name='max_bin'),
        # Real(0.1, 30, name='min_child_weight'),
        # Real(0, 30, name= 'min_split_gain'),
        # Integer(2, 300, name='min_child_samples'),
        # Real(0.3, 1.0, name='colsample_bytree'),
        # Real(1e-10, 1e-2, 'log-uniform',name='reg_lambda'),
        # Real(1e-10, 1e-2, 'log-uniform', name='reg_alpha'),
        # Real(1.0, 500.0, 'uniform', name='scale_pos_weight'),
        ],
    'CatBoost' : [
        Real(0.001, 0.5, 'log-uniform', name='learning_rate'), 
        Integer(2, 10, name='depth'),
        Integer(100, 1000, name= 'n_estimators'),                   # Number of boosted trees to fit
        Real(0.3, 1.0, name='colsample_bylevel'),
        Integer(5, 40, name='one_hot_max_size'),    
        Categorical(['Buckets','Borders','BinarizedTargetMeanValue','Counter'],name='simple_ctr'),
        Categorical(['Buckets','Borders','BinarizedTargetMeanValue','Counter'],name='combinations_ctr'),
        ]
        }


def tune(default_params, space_name, X_train, y_train, optimizer, checkpointname, folds=(5,5), niter=100, rstate=42):
    """Bayesian parameter tuning for a given SkLearn classifier 

    Args:
        clf (SkLearn classifier): Model or pipeline to tune params
        space_name (str): Parameter search space name
        X_train (np.2darray): Input features
        y_train (np.1darray): Labels
        optimizer (str): Choice of optimizer, either 'gp' or 'gbrt'
        checkpointname (str): Filename to save checkpoint
        folds (tuple, optional): Do i iterations of k folds as (i,k) for faster evaluation. 
        niter (int, optional): Number of steps for optimization.
        rstate (int, optional): Randomness seed. 

    Returns:
        scipy.optimize.OptimizeResult: skopt optimization result
    """        


    space = SPACES[space_name]

    os.makedirs('./results', exist_ok=True)

    # this decorator allows your objective function to receive a the parameters as
    # keyword arguments. This is particularly convenient when you want to set
    # scikit-learn estimator parameters
    @use_named_args(space)
    def objective(**params):

        clf = get_model(name=space_name, params={**default_params,**params})

        _, loss, _ = validate_model(clf, X_train, y_train, folds, rstate, verbose=False )
        print(f'Tuner loss: {round(loss,4)} for params: {params}')
        
        return loss

    if optimizer=='gp':
        opt_fun = gp_minimize
    elif optimizer=='gbrt':
        opt_fun = gbrt_minimize
    else:
        opt_fun = gp_minimize
    
    res = opt_fun(objective, 
                space, 
                n_calls=niter, 
                random_state=rstate,
                verbose=False,
                callback=[
                    DeltaYStopper(delta=0.002, n_best=10), 
                    CheckpointSaver(f"./results/{checkpointname}.pkl", store_objective=False)],
                )


    dump(res, f'./results/{checkpointname}_final.pkl', store_objective=False)

    optimum_params = {p.name:v for p,v in zip(res.space, res.x)}

    return res, optimum_params