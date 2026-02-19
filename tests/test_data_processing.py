"""
Unit tests for data_processing module.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_processing import (
    load_data,
    preprocess_data,
    split_features_target,
    save_processed_data,
)


class TestLoadData:
    """Tests for load_data function."""

    def test_load_csv_file(self, tmp_path):
        """Test loading CSV file."""
        # Create a temporary CSV file
        test_file = tmp_path / "test_data.csv"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.to_csv(test_file, index=False)

        # Load the file
        loaded_df = load_data(test_file)

        # Assertions
        assert isinstance(loaded_df, pd.DataFrame)
        assert loaded_df.shape == (3, 2)
        pd.testing.assert_frame_equal(loaded_df, df)

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_data(Path("nonexistent_file.csv"))

    def test_load_unsupported_format(self, tmp_path):
        """Test loading an unsupported file format."""
        test_file = tmp_path / "test_data.txt"
        test_file.write_text("some text")

        with pytest.raises(ValueError):
            load_data(test_file)


class TestPreprocessData:
    """Tests for preprocess_data function."""

    def test_preprocess_with_target(self):
        """Test preprocessing with target column."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": [0, 1, 0, 1, 0]
        })

        X, y = preprocess_data(df, target_col="target", scale_features=False)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape == (5, 2)
        assert len(y) == 5
        assert "target" not in X.columns

    def test_preprocess_without_target(self):
        """Test preprocessing without target column."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10]
        })

        result = preprocess_data(df, scale_features=False)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 2)

    def test_preprocess_drop_columns(self):
        """Test dropping specific columns."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "feature3": [7, 8, 9]
        })

        result = preprocess_data(df, drop_cols=["feature3"], scale_features=False)

        assert result.shape == (3, 2)
        assert "feature3" not in result.columns

    def test_preprocess_handle_missing_drop(self):
        """Test handling missing values by dropping."""
        df = pd.DataFrame({
            "feature1": [1, 2, np.nan, 4],
            "feature2": [2, 4, 6, 8]
        })

        result = preprocess_data(df, handle_missing="drop", scale_features=False)

        assert result.shape == (3, 2)
        assert not result.isnull().any().any()

    def test_preprocess_scaling(self):
        """Test feature scaling."""
        df = pd.DataFrame({
            "feature1": [10, 20, 30, 40, 50],
            "feature2": [100, 200, 300, 400, 500]
        })

        result = preprocess_data(df, scale_features=True)

        # Check that mean is close to 0 and std is close to 1
        assert abs(result["feature1"].mean()) < 1e-10
        assert abs(result["feature2"].mean()) < 1e-10
        assert abs(result["feature1"].std() - 1.0) < 0.1
        assert abs(result["feature2"].std() - 1.0) < 0.1


class TestSplitFeaturesTarget:
    """Tests for split_features_target function."""

    def test_split_valid_target(self):
        """Test splitting with valid target column."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })

        X, y = split_features_target(df, "target")

        assert X.shape == (3, 2)
        assert len(y) == 3
        assert "target" not in X.columns

    def test_split_invalid_target(self):
        """Test splitting with invalid target column."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6]
        })

        with pytest.raises(ValueError):
            split_features_target(df, "nonexistent")


class TestSaveProcessedData:
    """Tests for save_processed_data function."""

    def test_save_csv(self, tmp_path):
        """Test saving data to CSV."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        output_file = tmp_path / "output.csv"

        save_processed_data(df, output_file)

        assert output_file.exists()
        loaded_df = pd.read_csv(output_file)
        pd.testing.assert_frame_equal(loaded_df, df)

    def test_save_creates_directory(self, tmp_path):
        """Test that save creates parent directories if they don't exist."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        output_file = tmp_path / "subdir" / "output.csv"

        save_processed_data(df, output_file)

        assert output_file.exists()


# Pytest fixtures
@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame for testing."""
    return pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.randn(100),
        "target": np.random.choice([0, 1], 100)
    })
