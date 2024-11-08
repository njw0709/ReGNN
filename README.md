# Regression Guided Neural Networks (ReGNN)

This repository contains code for the paper "Unveiling Population Heterogeneity in Health Risks Posed by Environmental Hazards Using Regression-Guided Neural Network".

## Setup

1. Create a python environment with all of the dependencies installed

```bash
conda create -n regnn python=3.9
conda activate regnn
pip install -r requirements.txt
```

2. We are using stata to run the regression models. You can install stata from [here](https://www.stata.com/).

3. Change code in hyperparam/eval.py to point to the stata executable on your machine.
   Example:

```python
def init_stata():
    stata_setup.config("/usr/local/stata17", "mp") # Change this to point to your stata executable
```

4. Install mihm library in your python environment. You need to be in the base directory of this repository.

```bash
pip install -e . #install mihm in editable mode
```

## Experiments

Our main experiment that assesses the effect of air pollution on cognition can be found under "notebook" directory:

- [CognitionAirPollutionExample.ipynb](notebook/CognitionAirPollutionExample.ipynb)
