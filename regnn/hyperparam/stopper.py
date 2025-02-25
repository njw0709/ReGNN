from ray import tune
import numpy as np


class NaNStopper(tune.Stopper):
    def __init__(self, metric="loss"):
        self.metric = metric

    def __call__(self, trial_id, result):
        # Check if the current loss is NaN
        return result.get(self.metric, None) is None or np.isnan(result[self.metric])

    def stop_all(self):
        return False
