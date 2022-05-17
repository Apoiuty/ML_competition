import numpy as np
from sklearn.ensemble import AdaBoostRegressor
import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_log_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from ray.tune.sklearn import TuneSearchCV
from sklearn.tree import DecisionTreeRegressor
import torch
from ray import tune

np.random.seed(42)
all_features, y = torch.load('../input/housing-price/house_price.pkl')
X = all_features.iloc[:len(y), :]
X, y = X.values, y.values
X_test = all_features.iloc[len(y):, :]

model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor())
param = {
    'learning_rate': tune.loguniform(1e-1, 1),
    'base_estimator__max_depth': tune.randint(3, 100),
    'base_estimator__min_samples_split': tune.randint(2, len(X) + 1),
    'base_estimator__min_samples_leaf': tune.randint(1, len(X) + 1),
    'base_estimator__max_features': tune.randint(2, X.shape[1]),
    'base_estimator__max_leaf_nodes': tune.randint(1, len(X) + 1),
    'n_estimators': tune.randint(10, 1000)
}
tune_search = TuneSearchCV(
    model,
    param,
    search_optimization="bayesian",
    n_trials=-1,
    max_iters=10,
    early_stopping=True,
    verbose=1,
    scoring=make_scorer(mean_absolute_error, greater_is_better=False),
    return_train_score=True,
    mode='max',
    time_budget_s=3600 * 2,
    error_score=np.nan,
    use_gpu=True
)
tune_search.fit(X, y)
print(tune_search.best_params_)
