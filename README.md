# germanHousing

- Feature exploratory analysis can be found in [EDA notebook](housing-eda.ipynb).
- Model Performance Analysis, in `train_nested_CV.ipynb`

## What's Done

### Preprocessing

### Model Selection and Training
For model selection and parameter tuning, employed nested cross validation with Bayesian Optimization Library [SkOpt](https://scikit-optimize.github.io/stable/index.html).

    
## Improvements 
- [ ] subtract `yearConstructed` and `lastRefurbish` from scraping `date` to make these columns reflect recency relative to listing date  
- [ ] finetune a popular transformer for German text embeddings
- [ ] explore other embedding and dimension reduction strategies such as Umap, PCA, Auto Encoders
- [ ] explore feature importances using CatBoost's own modules and eliminate redundant ones
