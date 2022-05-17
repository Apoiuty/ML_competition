import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from ray import tune

all_features, y = torch.load('house_price.pkl')
X = all_features.iloc[:len(y), :]
X, y = X.values, y.values

pca = PCA(n_components=0.5)
tree = RandomForestRegressor(n_estimators=100, oob_score=True)
pipeline = make_pipeline(pca, tree)
print(pipeline)
params = {'randomforestregressor__n_estimators': np.arange(100, 201, 10).tolist()}


grid_search = GridSearchCV(pipeline, params, scoring=make_scorer(mean_squared_log_error, greater_is_better=False),
                           cv=5,
                           verbose=2, n_jobs=-1)
grid_search.fit(X, y)
print(grid_search.best_params_)



