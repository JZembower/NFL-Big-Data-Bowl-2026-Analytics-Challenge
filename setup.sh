#!/bin/bash

# NFL Big Data Bowl 2026 - Setup Script
# This script sets up the project structure and environment

echo "=========================================="
echo "NFL Big Data Bowl 2026 - Project Setup"
echo "=========================================="

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/splits
mkdir -p checkpoints
mkdir -p logs
mkdir -p predictions
mkdir -p visualizations
mkdir -p notebooks

echo "✓ Directories created"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✓ Dependencies installed"

# Create .gitignore
echo ""
echo "Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Data
data/raw/*.csv
data/processed/*.parquet
data/splits/*.parquet

# Models
checkpoints/*.pkl
checkpoints/*.pth

# Logs
logs/*.log
logs/*.txt

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Outputs
predictions/*.csv
visualizations/*.png
visualizations/*.jpg
EOF

echo "✓ .gitignore created"

# Create placeholder notebooks
echo ""
echo "Creating placeholder notebooks..."

cat > notebooks/01_eda.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# NFL Big Data Bowl 2026 - Exploratory Data Analysis\n",
    "\n",
    "This notebook contains exploratory analysis of the tracking data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('../src')\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from data_loader import NFLDataLoader\n",
    "\n",
    "%matplotlib inline\n",
    "%load_ext autoreload\n",
    "%autoreload 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data\n",
    "loader = NFLDataLoader()\n",
    "input_df, output_df = loader.load_week_data(1)\n",
    "supp_df = loader.load_supplementary_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Your EDA code here\n",
    "input_df.head()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

cat > notebooks/02_features.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# NFL Big Data Bowl 2026 - Feature Engineering\n",
    "\n",
    "This notebook experiments with different feature engineering approaches."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('../src')\n",
    "\n",
    "from data_loader import NFLDataLoader\n",
    "from feature_engineering import FeatureEngineer\n",
    "\n",
    "%load_ext autoreload\n",
    "%autoreload 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Your feature engineering experiments here"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

cat > notebooks/03_modeling.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# NFL Big Data Bowl 2026 - Modeling Experiments\n",
    "\n",
    "This notebook contains model training and evaluation experiments."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('../src')\n",
    "\n",
    "from models.baseline import BaselineModel\n",
    "from models.lstm import LSTMTrainer\n",
    "\n",
    "%load_ext autoreload\n",
    "%autoreload 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Your modeling experiments here"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo "✓ Placeholder notebooks created"

# Print success message
echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Place your CSV files in data/raw/"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the pipeline: python src/train.py --model xgboost"
echo ""
echo "For more information, see README.md"
echo ""