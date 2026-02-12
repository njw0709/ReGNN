# Regression Guided Neural Networks (ReGNN)

This repository contains code for the paper "Unveiling Population Heterogeneity in Health Risks Posed by Environmental Hazards Using Regression-Guided Neural Network".

## Requirements

- Python >= 3.9
- (Optional) [Stata](https://www.stata.com/) — only needed if you want to use the Stata-backed regression functions (`OLS_stata`, `VIF_stata`). The pure-Python alternative via **statsmodels** (`OLS_statsmodel`, `VIF_statsmodel`) works without Stata.

## Installation

### Option 1 — uv (recommended)

[uv](https://docs.astral.sh/uv/) is the recommended way to manage the environment.

```bash
# Create a virtual environment and install the package in editable mode
uv venv --python 3.9
source .venv/bin/activate
uv pip install -e .
```

To include development dependencies (pytest, black):

```bash
uv pip install -e ".[dev]"
```

### Option 2 — pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

With development extras:

```bash
pip install -e ".[dev]"
```

### Option 3 — conda + pip

```bash
conda create -n regnn python=3.9
conda activate regnn
pip install -e .
```

### Legacy requirements.txt

A `requirements.txt` file is still provided for backward compatibility, but `pyproject.toml` is the source of truth for dependencies.

```bash
pip install -r requirements.txt
pip install -e .
```

## Stata Configuration (optional)

If you plan to use the Stata-backed evaluation functions, install Stata and point the library to your local installation:

```python
from regnn.eval import init_stata

# Adjust the path and edition to match your Stata installation
init_stata()  # uses stata_setup.config("/usr/local/stata17", "mp") by default
```

If you do **not** have Stata, you can use the statsmodels equivalents instead — no extra configuration is needed:

```python
from regnn.eval import OLS_statsmodel, VIF_statsmodel
```

## Experiments

Our main experiment that assesses the effect of air pollution on cognition can be found under "notebook" directory:

- [CognitionAirPollutionExample.ipynb](notebook/CognitionAirPollutionExample.ipynb)

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- NonCommercial — You may not use the material for commercial purposes.
- ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

Full license text: https://creativecommons.org/licenses/by-nc-sa/4.0/


