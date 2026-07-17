"""
models.py
---------
Model training, hyperparameter tuning, and evaluation utilities.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from typing import Dict


def get_param_grid(model_type: str) -> Dict:
    """
    Get hyperparameter grid for GridSearchCV.

    Parameters
    ----------
    model_type : str
        'logistic_regression' or 'random_forest'.

    Returns
    -------
    Dict
        Parameter grid dictionary.
    """
    if model_type == 'logistic_regression':
        return {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga']
        }
    elif model_type == 'random_forest':
        return {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_with_gridsearch(pipeline, param_grid, X_train, y_train, 
                          cv_folds=5, random_state=42, scoring='roc_auc'):
    """
    Train a pipeline with GridSearchCV for hyperparameter tuning.

    Parameters
    ----------
    pipeline : Pipeline
        Scikit-learn pipeline.
    param_grid : dict
        Hyperparameter grid.
    X_train, y_train : array-like
        Training data.
    cv_folds : int
        Number of cross-validation folds.
    random_state : int
        Random seed.
    scoring : str
        Scoring metric for GridSearchCV.

    Returns
    -------
    GridSearchCV
        Fitted GridSearchCV object.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)
    return grid


def evaluate_model(grid_search, X_test, y_test, model_name: str) -> Dict:
    """
    Comprehensive model evaluation.

    Returns
    -------
    Dict
        Dictionary with all metrics.
    """
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    return {
        'model_name': model_name,
        'best_params': grid_search.best_params_,
        'cv_roc_auc': grid_search.best_score_,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'y_pred': y_pred,
        'y_prob': y_prob
    }
