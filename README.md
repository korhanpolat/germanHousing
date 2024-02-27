# germanHousing

- Model Selection and Tuning can be viewed in [`model_eval_nested_CV.ipynb`](model_eval_nested_CV.ipynb) notebook
- [`skopt_tune.py`](skopt_tune.py) : Employs Bayesian hyperparameter tuning
- [`train_utils.py`](train_utils.py) : Inner loop of nested cross validation
- [`preprocess.py`](preprocess.py) : Feature preprocessing and transformations
- [`model_config.py`](model_config.py) : Model getter

## What's Done

- Each feature is examined and some irrelevant features are discarded, as can be found in [`housing_eda.ipynb`](housing_eda.ipynb)
- Used [CatBoost](https://catboost.ai/) as the main **Regressor model**, because it handles categorical variables with sophisticated algorithms and can handle N/A's as 'other' class.
  It also supports text features and embedding features. 
- **Target** is chosen as `totalRent`. Missing or abnormal values are treated according to this relation:
  $$assumedRent = baseRent + serviceCharge + heatingCosts$$
  Also $log(rent)$ is used since it's closer to Normal distribution.
- For **model selection and parameter tuning**, employed nested cross validation with Bayesian Optimization Library [SkOpt](https://scikit-optimize.github.io/stable/index.html).
  Tuned CatBoost model without text features and achieved $RMSE:0.14$ and $R^{2}:0.91$ for log prices.
    <img width="400" alt="resim" src="https://github.com/korhanpolat/germanHousing/assets/25014836/65eb7004-92ac-41cc-bf8f-78c54d1c409a">



- Encoded text embeddings with [Sentence Transformers](https://www.sbert.net/) by using a Multilingual model that supports German.

*Did not have time to tune for encoded embeddings and text features.*


    
## Improvements 
- [ ] Tune with text embeddings
- [ ] Use Bag of Words n-grams features for text columns 
- [ ] Explore other embedding and dimension reduction strategies such as Umap, PCA, Auto Encoders
- [ ] Explore feature importances using CatBoost's own modules and eliminate redundant ones
- [ ] Subtract `yearConstructed` and `lastRefurbish` from scraping `date` to make these columns reflect recency relative to listing date  
- [ ] Finetune a popular transformer for German text embeddings
      
