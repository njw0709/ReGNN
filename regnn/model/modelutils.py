import torch
from sklearn.base import BaseEstimator, RegressorMixin
import inspect
import pandas as pd


class SklearnCompatibleModel(BaseEstimator, RegressorMixin):
    def __init__(self, model, device="cpu"):
        super(BaseEstimator, self).__init__()
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
        self._model = model
        self._model.eval()
        self.device = device
        if self.model.device != self.device:
            self.model.to(self.device)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            output = self.model(X)
        if self.device == "cpu":
            return output.numpy().reshape(-1, 1)
        else:
            return output.detach().cpu().numpy().reshape(-1, 1)

    def predict_above(self, X, val=0.9, above: bool = True):
        outcome = self.predict(X)
        if above:
            outcome_bin = outcome > val
        else:
            outcome_bin = outcome < val
        return outcome_bin

    def fit(self, X, y):
        self.is_fitted_ = True
        return self
