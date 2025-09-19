import numpy as np
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any

class Utils:
    
    @staticmethod
    def get_min(values: list) -> float:
        if len(values) == 0:
            return float('nan')
        
        tmp = values[0]
        for value in values:
            if value < tmp:
                tmp = value
        return float(tmp)
    
    @staticmethod
    def get_max(values: list) -> float:
        if len(values) == 0:
            return float('nan')
        
        tmp = values[0]
        for value in values:
            if value > tmp:
                tmp = value
        return float(tmp)

    @staticmethod    
    def get_numeric_features(df: pd.DataFrame) -> List[str]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Index' in numeric_cols:
            numeric_cols.remove('Index')
        return numeric_cols
    
    @staticmethod
    def create_output_directory(output_dir : str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)