"""
data_loader.py
--------------
Data ingestion, cleaning, and feature type identification for Telco Customer Churn.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


def load_telco_data(url: str = None) -> pd.DataFrame:
    """
    Load the IBM Telco Customer Churn dataset.

    Parameters
    ----------
    url : str, optional
        Custom URL. Defaults to IBM's public GitHub source.

    Returns
    -------
    pd.DataFrame
        Raw dataset with 7,043 rows and 21 columns.
    """
    if url is None:
        url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'

    df = pd.read_csv(url)
    return df


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Telco dataset:
    - Drop CustomerID
    - Convert TotalCharges to numeric
    - Encode target variable

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for modeling.
    """
    df = df.copy()
    df = df.drop('customerID', axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df


def get_feature_types(df: pd.DataFrame, target_col: str = 'Churn') -> Tuple[List[str], List[str]]:
    """
    Automatically identify numeric and categorical feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    target_col : str
        Name of the target column to exclude.

    Returns
    -------
    Tuple[List[str], List[str]]
        (numeric_features, categorical_features)
    """
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_col in numeric_features:
        numeric_features.remove(target_col)

    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    return numeric_features, categorical_features
