"""
Visualization Module

This module contains functions for creating plots and visualizations.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


def plot_correlation_matrix(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (12, 10),
    cmap: str = "coolwarm",
    annot: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot correlation matrix heatmap.

    Args:
        df: Input DataFrame
        figsize: Figure size as (width, height)
        cmap: Colormap for the heatmap
        annot: Whether to annotate cells with correlation values
        save_path: Path to save the figure (optional)

    Example:
        >>> plot_correlation_matrix(df, figsize=(10, 8))
    """
    # Calculate correlation matrix
    corr = df.select_dtypes(include=[np.number]).corr()

    # Create figure
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        cmap=cmap,
        annot=annot,
        fmt=".2f" if annot else "",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Matrix", fontsize=16, pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved correlation matrix to {save_path}")

    plt.show()


def plot_pca_variance(
    pca: PCA,
    figsize: tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot explained variance by PCA components.

    Args:
        pca: Fitted PCA object
        figsize: Figure size as (width, height)
        save_path: Path to save the figure (optional)

    Example:
        >>> from sklearn.decomposition import PCA
        >>> pca = PCA().fit(X)
        >>> plot_pca_variance(pca)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Individual variance explained
    ax1.bar(
        range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_
    )
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained")
    ax1.set_title("Variance Explained by Each Component")
    ax1.grid(True, alpha=0.3)

    # Cumulative variance explained
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, len(cumsum) + 1), cumsum, marker="o")
    ax2.axhline(y=0.95, color="r", linestyle="--", label="95% threshold")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance Explained")
    ax2.set_title("Cumulative Variance Explained")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved PCA variance plot to {save_path}")

    plt.show()


def plot_feature_distributions(
    df: pd.DataFrame,
    features: Optional[list[str]] = None,
    n_cols: int = 3,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot distributions of features.

    Args:
        df: Input DataFrame
        features: List of feature names to plot (if None, plots all numeric features)
        n_cols: Number of columns in the subplot grid
        figsize: Figure size as (width, height)
        save_path: Path to save the figure (optional)

    Example:
        >>> plot_feature_distributions(df, features=['age', 'income', 'score'])
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (n_cols * 4, n_rows * 3)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(features):
        if feature in df.columns:
            df[feature].hist(bins=30, ax=axes[idx], edgecolor="black")
            axes[idx].set_title(f"Distribution of {feature}")
            axes[idx].set_xlabel("Value")
            axes[idx].set_ylabel("Frequency")

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved feature distributions to {save_path}")

    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Optional[list[str]] = None,
    figsize: tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix array
        labels: Class labels
        figsize: Figure size as (width, height)
        save_path: Path to save the figure (optional)

    Example:
        >>> from sklearn.metrics import confusion_matrix
        >>> cm = confusion_matrix(y_true, y_pred)
        >>> plot_confusion_matrix(cm, labels=['Class 0', 'Class 1'])
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.title("Confusion Matrix", fontsize=16)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved confusion matrix to {save_path}")

    plt.show()
