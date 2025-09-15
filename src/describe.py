import pandas as pd
import numpy as np
from typing import List, Dict, Any

class Describe():
    
    def __init__(self, filepath: str):
        self.df = pd.read_csv(filepath)
        
    def _truncate_name(self, name: str, max_len: int = 15) -> str:
        if len(name) <= max_len:
            return name
        return name[:max_len-3] + "..."

    def print_data(self):
        numeric_features = self._get_numeric_features()
        col_width = 14
        
        print(f"{'':>10}", end="")
        for feature in numeric_features:
            truncated = self._truncate_name(feature, col_width-2)
            print(f"{truncated:>{col_width}}", end="")
        print()
        
        stats_names = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
        
        for stat_name in stats_names:
            print(f"{stat_name:>10}", end="")
            for feature in numeric_features:
                stats = self._calculate_stats(feature)
                print(f"{stats[stat_name]:>{col_width}.6f}", end="")
            print()

    def _calculate_stats(self, feature_name: str) -> Dict[str, float]:
        return {
            'Count': self._get_count(feature_name),
            'Mean': self._get_mean(feature_name),
            'Std': self._get_std(feature_name),
            'Min': self._get_min(feature_name),
            '25%': self._get_percentile(feature_name, 25),
            '50%': self._get_percentile(feature_name, 50),
            '75%': self._get_percentile(feature_name, 75),
            'Max': self._get_max(feature_name)
        }
    
    def _get_numeric_features(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numeric_cols if col != 'Index']
    
    def _get_count(self, feature_name: str) -> float:
        column_data = self.df[feature_name].dropna()
        return float(len(column_data))

    def _get_mean(self, feature_name: str) -> float:
        column_data = self.df[feature_name].dropna()
        if len(column_data) == 0:
            return float('nan')
        return sum(column_data) / len(column_data)
    
    def _get_std(self, feature_name: str) -> float:
        column_data = self.df[feature_name].dropna()
        if len(column_data) <= 1:
            return float('nan')
        
        mean = self._get_mean(feature_name)
        variance = sum((x - mean) ** 2 for x in column_data) / (len(column_data) - 1)
        return variance ** 0.5
    
    def _get_min(self, feature_name: str) -> float:
        column_data = self.df[feature_name].dropna()
        if len(column_data) == 0:
            return float('nan')
        return float(min(column_data))
    
    def _get_max(self, feature_name: str) -> float:
        column_data = self.df[feature_name].dropna()
        if len(column_data) == 0:
            return float('nan')
        return float(max(column_data))
    
    def _get_percentile(self, feature_name: str, percentile: float) -> float:
        column_data = self.df[feature_name].dropna()
        if len(column_data) == 0:
            return float('nan')
        
        sorted_values = sorted(column_data)
        n = len(sorted_values)
        
        index = (percentile / 100.0) * (n - 1)
        
        if index.is_integer():
            return float(sorted_values[int(index)])
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            
            if upper_index >= n:
                return float(sorted_values[-1])
            
            lower_value = sorted_values[lower_index]
            upper_value = sorted_values[upper_index]
            
            weight = index - lower_index
            return float(lower_value + weight * (upper_value - lower_value))

def main():
    describe = Describe("datasets/dataset_train.csv")
    describe.print_data()

if __name__ == "__main__":
    main()