# Project Structure

Complete repository structure for CDA Case 1: The High-Dimensional Standoff

```
cda-case1/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD pipeline (lint, test)
├── data/
│   ├── raw/
│   │   └── .gitkeep              # Git tracking for raw data directory
│   └── processed/
│       └── .gitkeep              # Git tracking for processed data directory
├── docs/
│   ├── index.md                  # Documentation homepage
│   └── mkdocs.yml                # MkDocs configuration
├── notebooks/
│   └── analysis.ipynb            # Main analysis Jupyter notebook
├── src/
│   ├── __init__.py               # Package initialization with exports
│   ├── data_processing.py        # Data loading and preprocessing functions
│   ├── models.py                 # Machine learning model functions
│   └── visualization.py          # Plotting and visualization functions
├── tests/
│   ├── __init__.py               # Test package initialization
│   └── test_data_processing.py   # Unit tests for data_processing module
├── .gitignore                    # Git ignore rules (Python, Jupyter, data files)
├── LICENSE                       # MIT License
├── README.md                     # Comprehensive project documentation
├── environment.yml               # Conda environment specification
├── pyproject.toml                # Project metadata and uv/pip dependencies
└── requirements.txt              # Pip requirements (fallback)
```

## Key Features

✨ **Modern Tooling**
- uv for fast dependency management (via pyproject.toml)
- Fallback support for conda (environment.yml) and pip (requirements.txt)

📊 **Complete Data Science Stack**
- numpy, pandas for data manipulation
- matplotlib, seaborn for visualization
- scikit-learn for machine learning
- jupyterlab with ipykernel for interactive analysis

🧪 **Development Tools**
- black for code formatting
- ruff for fast linting
- pytest with coverage for testing

📚 **Documentation**
- mkdocs with material theme
- Comprehensive README with badges and instructions
- API documentation support

🤖 **CI/CD**
- GitHub Actions workflow
- Automated linting and testing
- Multi-version Python testing (3.9, 3.10, 3.11)

## Installation Commands

### Using uv (Recommended)
```bash
uv sync
```

### Using conda
```bash
conda env create -f environment.yml
conda activate cda-case1
```

### Using pip
```bash
pip install -r requirements.txt
```

## Development Commands

### Run Tests
```bash
uv run pytest
```

### Format Code
```bash
uv run black src/ tests/
```

### Lint Code
```bash
uv run ruff check src/ tests/
```

### Start Jupyter Lab
```bash
uv run jupyter lab
```

### Build Documentation
```bash
cd docs && uv run mkdocs serve
```
