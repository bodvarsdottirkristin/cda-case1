# CDA Case 1: The High-Dimensional Standoff

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> Repository for Case 1: The High-Dimensional Standoff in the course Computational Data Analysis (02582)

## 📋 Overview

This project contains a complete data science workflow for analyzing high-dimensional data. The main analysis is contained in the Jupyter notebook `analysis.ipynb`, with supporting modules for data processing, visualization, and modeling.

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) (recommended), conda, or pip

### Installation

#### Option 1: Using uv (Recommended)

uv is a modern, fast Python package manager. Install it first if you don't have it:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then sync dependencies:

```bash
uv sync
```

#### Option 2: Using conda

```bash
conda env create -f environment.yml
conda activate cda-case1
```

#### Option 3: Using pip

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📊 Usage

### Running the Analysis

1. Start JupyterLab:
   ```bash
   # With uv
   uv run jupyter lab
   
   # With conda/pip
   jupyter lab
   ```

2. Open `notebooks/analysis.ipynb` and run the cells sequentially.

### Project Structure

```
cda-case1/
├── data/
│   ├── raw/                # Original, immutable data
│   └── processed/          # Cleaned, transformed data
├── docs/                   # Documentation (MkDocs)
│   ├── index.md
│   └── mkdocs.yml
├── notebooks/              # Jupyter notebooks
│   └── analysis.ipynb      # Main analysis notebook
├── src/                    # Source code for reusable modules
│   ├── __init__.py
│   ├── data_processing.py  # Data loading and preprocessing
│   ├── visualization.py    # Plotting utilities
│   └── models.py           # Machine learning models
├── tests/                  # Unit tests
│   ├── __init__.py
│   └── test_data_processing.py
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
├── README.md               # This file
├── environment.yml         # Conda environment specification
├── pyproject.toml          # Project metadata and dependencies (uv/pip)
└── requirements.txt        # Pip requirements (fallback)
```

## 🧪 Development

### Code Quality

The project uses [black](https://github.com/psf/black) for formatting and [ruff](https://github.com/astral-sh/ruff) for linting.

```bash
# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/

# Fix auto-fixable issues
uv run ruff check --fix src/ tests/
```

### Testing

Run tests with pytest:

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src --cov-report=html
```

### Building Documentation

```bash
# Serve documentation locally
uv run mkdocs serve

# Build documentation
uv run mkdocs build
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linters
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Course: Computational Data Analysis (02582)
- Institution: Technical University of Denmark

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
