"""
CDA Case 1 - Source Package

This package contains reusable modules for data processing,
visualization, and modeling.
"""

__version__ = "0.1.0"
__author__ = "bodvarsdottirkristin"

# Import main functions for convenience
from .data_processing import load_data, preprocess_data
from .models import evaluate_model, train_model
from .visualization import plot_correlation_matrix, plot_pca_variance

__all__ = [
    "load_data",
    "preprocess_data",
    "plot_correlation_matrix",
    "plot_pca_variance",
    "train_model",
    "evaluate_model",
]
