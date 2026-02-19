"""
Data Processing Module

This module contains functions for loading, cleaning, and preprocessing data.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(filepath: Path, **kwargs) -> pd.DataFrame:
    """
    Load data from a file.

    Args:
        filepath: Path to the data file
        **kwargs: Additional arguments passed to pandas read function

    Returns:
        DataFrame containing the loaded data

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is not supported

    Example:
        >>> df = load_data(Path('data/raw/dataset.csv'))
        >>> print(df.shape)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Determine file type and load accordingly
    if filepath.suffix == ".csv":
        df = pd.read_csv(filepath, **kwargs)
    elif filepath.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath, **kwargs)
    elif filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath, **kwargs)
    elif filepath.suffix == ".json":
        df = pd.read_json(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    print(f"✓ Loaded data from {filepath.name}: {df.shape}")
    return df


def preprocess_data(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    drop_cols: Optional[list] = None,
    handle_missing: str = "drop",
    scale_features: bool = True,
) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Preprocess data by handling missing values, dropping columns, and scaling.

    Args:
        df: Input DataFrame
        target_col: Name of the target column (if present)
        drop_cols: List of columns to drop
        handle_missing: Strategy for handling missing values ('drop', 'mean', 'median')
        scale_features: Whether to scale features using StandardScaler

    Returns:
        Tuple of (processed_features, target) if target_col is provided,
        otherwise just processed_features

    Example:
        >>> X, y = preprocess_data(df, target_col='target', scale_features=True)
    """
    df = df.copy()

    # Drop specified columns
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    # Separate target if specified
    target = None
    if target_col and target_col in df.columns:
        target = df[target_col]
        df = df.drop(columns=[target_col])

    # Handle missing values
    if handle_missing == "drop":
        df = df.dropna()
        if target is not None:
            target = target[df.index]
    elif handle_missing == "mean":
        df = df.fillna(df.mean())
    elif handle_missing == "median":
        df = df.fillna(df.median())

    # Scale features
    if scale_features:
        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print(f"✓ Preprocessed data: {df.shape}")

    return (df, target) if target is not None else df


def split_features_target(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target.

    Args:
        df: Input DataFrame
        target_col: Name of the target column

    Returns:
        Tuple of (features, target)

    Example:
        >>> X, y = split_features_target(df, 'target')
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def save_processed_data(df: pd.DataFrame, filepath: Path) -> None:
    """
    Save processed data to file.

    Args:
        df: DataFrame to save
        filepath: Path where to save the file

    Example:
        >>> save_processed_data(df, Path('data/processed/cleaned.csv'))
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.suffix == ".csv":
        df.to_csv(filepath, index=False)
    elif filepath.suffix == ".parquet":
        df.to_parquet(filepath, index=False)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    print(f"✓ Saved processed data to {filepath}")
