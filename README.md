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

3. Change code in train/eval.py to point to the stata executable on your machine.
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


