"""
Machine Learning Models Module

This module contains functions for training and evaluating models.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
    **model_params,
) -> BaseEstimator:
    """
    Train a machine learning model.

    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model to train ('random_forest', 'logistic', etc.)
        **model_params: Additional parameters for the model

    Returns:
        Trained model

    Example:
        >>> model = train_model(X_train, y_train, model_type='random_forest', n_estimators=100)
    """
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth", 10),
            random_state=model_params.get("random_state", 42),
            n_jobs=model_params.get("n_jobs", -1),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f"Training {model_type} model...")
    model.fit(X_train, y_train)
    print("✓ Model training complete")

    return model


def evaluate_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    print_report: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained model.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        print_report: Whether to print classification report

    Returns:
        Dictionary containing evaluation metrics

    Example:
        >>> metrics = evaluate_model(model, X_test, y_test)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    if print_report:
        print("\n" + "=" * 50)
        print("MODEL EVALUATION RESULTS")
        print("=" * 50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("=" * 50 + "\n")

    return metrics


def cross_validate_model(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "accuracy",
) -> Tuple[float, float]:
    """
    Perform cross-validation on a model.

    Args:
        model: Model to evaluate
        X: Features
        y: Labels
        cv: Number of cross-validation folds
        scoring: Scoring metric

    Returns:
        Tuple of (mean_score, std_score)

    Example:
        >>> mean_acc, std_acc = cross_validate_model(model, X, y, cv=5)
        >>> print(f"CV Accuracy: {mean_acc:.3f} (+/- {std_acc:.3f})")
    """
    print(f"Performing {cv}-fold cross-validation...")
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    mean_score = scores.mean()
    std_score = scores.std()

    print(f"✓ Cross-validation complete")
    print(f"{scoring.capitalize()} scores: {scores}")
    print(f"Mean {scoring}: {mean_score:.4f} (+/- {std_score:.4f})")

    return mean_score, std_score


def get_feature_importance(
    model: BaseEstimator,
    feature_names: Optional[list] = None,
    top_n: int = 20,
) -> Dict[str, float]:
    """
    Get feature importance from a trained model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        Dictionary of feature names and their importance scores

    Example:
        >>> importances = get_feature_importance(model, feature_names=X.columns)
        >>> for feat, imp in list(importances.items())[:5]:
        ...     print(f"{feat}: {imp:.4f}")
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have feature_importances_ attribute")

    importances = model.feature_importances_

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    # Create dictionary and sort by importance
    importance_dict = dict(zip(feature_names, importances))
    importance_dict = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )

    return importance_dict
