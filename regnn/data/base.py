from typing import Sequence, Callable, Union, Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

numeric = Union[int, float, complex, np.number]

class DatasetConfig(BaseModel):
    """Configuration for ReGNN datasets"""
    focal_predictor: str
    controlled_predictors: List[str]
    moderators: Union[List[str], List[List[str]]]
    outcome: str
    survey_weights: Optional[str] = None
    
class PreprocessStep(BaseModel):
    """Represents a preprocessing step with columns and function"""
    columns: List[str]
    function: Callable
    
    class Config:
        arbitrary_types_allowed = True

class BaseDataset:
    """Base class for dataset operations"""
    
    def __init__(self, df: pd.DataFrame, config: DatasetConfig):
        self.df = df
        self.columns = df.columns.tolist()
        self.config = config
        self.mean_std_dict: Dict[str, Tuple[float, float]] = {}
    
    def __len__(self) -> int:
        return len(self.df)
        
    def dropna(self, inplace: bool = True):
        if inplace:
            self.df = self.df.dropna()
        else:
            return self.df.dropna()
            
    def get_column_index(self, colname: str) -> int:
        return self.columns.index(colname) 