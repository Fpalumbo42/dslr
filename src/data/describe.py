import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import Utils
import pandas as pd
import numpy as np
import os
from typing import Dict

class Describe():
    
    def __init__(self, filepath: str) -> None:
        self.df = pd.read_csv(filepath)
        
    def _truncate_name(self, name: str, max_len: int = 15) -> str:
        if len(name) <= max_len:
            return name
        return name[:max_len-3] + "..."

    def _calculate_optimal_column_width(self, feature_count: int) -> int:
        terminal_width = os.get_terminal_size().columns
        label_width = 10
        available_width = terminal_width - label_width - 2
        
        if feature_count == 0:
            return 15
        
        optimal_width = available_width // feature_count
        return max(12, min(optimal_width, 20))

    def print_data(self) -> None:
        numeric_features = Utils.get_numeric_features(self.df)        
        col_width = self._calculate_optimal_column_width(len(numeric_features))
        
        print(f"{'':>10}", end="")
        for feature in numeric_features:
            truncated = self._truncate_name(feature, col_width-1)
            print(f"{truncated:>{col_width}}", end="")
        print()

        stats_names = ['Count', 'Mean', 'Std', 'Min', '10%', '25%', '50%', '75%', '90%', 'Max', 
                       'Range', 'IQR', 'Variance', 'Skewness', 'Kurtosis', 'MAD', 'Outliers']
        
        for stat_name in stats_names:
            print(f"{stat_name:>10}", end="")
            for feature in numeric_features:
                stats = self._calculate_stats(feature)
                value = stats[stat_name]
                formatted_value = self._format_value(value, col_width)
                print(f"{formatted_value:>{col_width}}", end="")
            print()

    def _format_value(self, value: float, col_width: int) -> str:
        if np.isnan(value):
            return "NaN"
        
        if col_width <= 12:
            if abs(value) >= 1000000:
                return f"{value:.1e}"
            elif abs(value) >= 1000:
                return f"{value:.0f}"
            elif abs(value) >= 1:
                return f"{value:.2f}"
            else:
                return f"{value:.3f}"
        else:
            if abs(value) >= 1000000:
                return f"{value:.2e}"
            elif abs(value) >= 1000:
                return f"{value:.1f}"
            elif abs(value) >= 1:
                return f"{value:.3f}"
            else:
                return f"{value:.4f}"

    def _calculate_stats(self, feature_name: str) -> Dict[str, float]:
        return {
            'Count': self._get_count(feature_name),
            'Mean': Utils.get_mean(self.df[feature_name].dropna()),
            'Std': Utils.get_std(self.df[feature_name].dropna()),
            'Min': Utils.get_min(self.df[feature_name].dropna()),
            '10%': self._get_percentile(feature_name, 10),
            '25%': self._get_percentile(feature_name, 25),
            '50%': self._get_percentile(feature_name, 50),
            '75%': self._get_percentile(feature_name, 75),
            '90%': self._get_percentile(feature_name, 90),
            'Max': Utils.get_max(self.df[feature_name].dropna()),
            'Range': self._get_range(feature_name),
            'IQR': self._get_iqr(feature_name),
            'Variance': self._get_variance(feature_name),
            'Skewness': self._get_skewness(feature_name),
            'Kurtosis': self._get_kurtosis(feature_name),
            'MAD': self._get_median_absolute_deviation(feature_name),
            'Outliers': self._get_outliers_count(feature_name)
        }

    def _get_count(self, feature_name: str) -> float:
        column_data = self.df[feature_name].dropna()
        len = 0
        for element in column_data:
            len += 1
        return float(len)

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

    def _get_variance(self, feature_name: str) -> float:
        std = Utils.get_std(self.df[feature_name].dropna())
        if pd.isna(std):
            return float('nan')
        return std ** 2

    def _get_range(self, feature_name: str) -> float:
        max_val = Utils.get_max(self.df[feature_name].dropna())
        min_val = Utils.get_min(self.df[feature_name].dropna())
        if pd.isna(max_val) or pd.isna(min_val):
            return float('nan')
        return max_val - min_val

    def _get_iqr(self, feature_name: str) -> float:
        q75 = self._get_percentile(feature_name, 75)
        q25 = self._get_percentile(feature_name, 25)
        if pd.isna(q75) or pd.isna(q25):
            return float('nan')
        return q75 - q25

    def _get_skewness(self, feature_name: str) -> float:
        column_data = self.df[feature_name].dropna()
        if len(column_data) < 3:
            return float('nan')
        
        mean = Utils.get_mean(self.df[feature_name].dropna())
        std = Utils.get_std(self.df[feature_name].dropna())
        if std == 0 or pd.isna(std):
            return float('nan')
        
        n = len(column_data)
        skew = sum(((x - mean) / std) ** 3 for x in column_data) / n
        return skew

    def _get_kurtosis(self, feature_name: str) -> float:
        column_data = self.df[feature_name].dropna()
        if len(column_data) < 4:
            return float('nan')
        
        mean = Utils.get_mean(self.df[feature_name].dropna())
        std = Utils.get_std(self.df[feature_name].dropna())
        if std == 0 or pd.isna(std):
            return float('nan')
        
        n = len(column_data)
        kurt = sum(((x - mean) / std) ** 4 for x in column_data) / n - 3
        return kurt

    def _get_median_absolute_deviation(self, feature_name: str) -> float:
        column_data = self.df[feature_name].dropna()
        if len(column_data) == 0:
            return float('nan')
        
        median = self._get_percentile(feature_name, 50)
        deviations = [abs(x - median) for x in column_data]
        sorted_deviations = sorted(deviations)
        n = len(sorted_deviations)
        
        if n % 2 == 0:
            return (sorted_deviations[n//2 - 1] + sorted_deviations[n//2]) / 2
        else:
            return sorted_deviations[n//2]

    def _get_outliers_count(self, feature_name: str) -> float:
        q25 = self._get_percentile(feature_name, 25)
        q75 = self._get_percentile(feature_name, 75)
        iqr = self._get_iqr(feature_name)
        
        if pd.isna(q25) or pd.isna(q75) or pd.isna(iqr):
            return float('nan')
        
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        column_data = self.df[feature_name].dropna()
        outliers = [x for x in column_data if x < lower_bound or x > upper_bound]
        return float(len(outliers))

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
        sys.exit(1)

    describe = Describe(sys.argv[1])
    describe.print_data()

if __name__ == "__main__":
    main()