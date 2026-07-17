"""
pipeline_builder.py
-------------------
Modular ML pipeline construction using scikit-learn Pipeline and ColumnTransformer.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer that handles numeric and categorical features separately.

    Parameters
    ----------
    numeric_features : List[str]
        List of numeric column names.
    categorical_features : List[str]
        List of categorical column names.

    Returns
    -------
    ColumnTransformer
        Fitted preprocessor pipeline.
    """
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor


def build_full_pipeline(preprocessor: ColumnTransformer, 
                        model_type: str = 'logistic_regression',
                        random_state: int = 42) -> Pipeline:
    """
    Build a complete ML pipeline: preprocessing + classifier.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Preprocessing transformer.
    model_type : str
        'logistic_regression' or 'random_forest'.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Pipeline
        Complete ML pipeline.
    """
    if model_type == 'logistic_regression':
        classifier = LogisticRegression(
            max_iter=1000, 
            random_state=random_state, 
            class_weight='balanced'
        )
    elif model_type == 'random_forest':
        classifier = RandomForestClassifier(
            random_state=random_state, 
            class_weight='balanced'
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return pipeline
