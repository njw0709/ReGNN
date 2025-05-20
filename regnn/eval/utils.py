import stata_setup
import torch
from regnn.model.regnn import ReGNN
import numpy as np
import pandas as pd
from regnn.constants import TEMP_DIR
import os
from typing import Union

# Track Stata initialization status
_stata_initialized = False


def init_stata():
    global _stata_initialized
    if not _stata_initialized:
        stata_setup.config("/usr/local/stata17", "mp")
        _stata_initialized = True
    from pystata import stata

    return stata
