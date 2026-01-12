import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import Utils
import pandas as pd
import matplotlib.pyplot as plt

class ScatterPlot:
    def __init__(self, filepath: str):
        self.df = pd.read_csv(filepath)
        self.houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        self.output_dir = "scatter_plots"
        Utils.create_output_directory(self.output_dir)
        self.plot_colors = {
            'Gryffindor': 'red',
            'Hufflepuff': 'yellow',
            'Ravenclaw': 'blue',
            'Slytherin': 'green'
        }
    
    def find_and_plot_most_similar_features(self):
        features = Utils.get_numeric_features(self.df)
        
        best_correlation = -1
        best_pair = None
        
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                feature1 = features[i]
                feature2 = features[j]
                
                correlation = self._calculate_correlation(feature1, feature2)
                
                if correlation is not None and abs(correlation) > abs(best_correlation):
                    best_correlation = correlation
                    best_pair = (feature1, feature2)
        
        if best_pair:
            print(f"Most similar features: {best_pair[0]} and {best_pair[1]}")
            print(f"Correlation coefficient: {best_correlation:.4f}")
            self.create_scatter(best_pair[0], best_pair[1], is_best=True)
        else:
            print("No valid feature pairs found")
    
    def _calculate_correlation(self, feature1: str, feature2: str):
        common_data = self._get_common_data(feature1, feature2)
        
        if len(common_data) < 2:
            return None
        
        x = common_data[feature1].values
        y = common_data[feature2].values
        
        n = len(x)
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        
        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        denominator = (sum_sq_x * sum_sq_y) ** 0.5
        
        if denominator == 0:
            return None
        
        return numerator / denominator
            
    def create_scatter(self, feature1: str, feature2: str, is_best: bool = False):
        common_data = self._get_common_data(feature1, feature2)
        
        if len(common_data) == 0:
            return
        
        plt.figure(figsize=(12, 10))
        
        for house in self.houses:
            house_data = common_data[common_data['Hogwarts House'] == house]
            
            if len(house_data) > 0:
                x = house_data[feature1]
                y = house_data[feature2]
                
                plt.scatter(x, y, c=self.plot_colors[house], label=house, alpha=0.6, edgecolor='black', linewidth=0.5, s=50)
        
        plt.xlabel(feature1, fontsize=12)
        plt.ylabel(feature2, fontsize=12)
        
        if is_best:
            plt.title(f'Most Similar Features: {feature1} vs {feature2}', fontsize=14, fontweight='bold')
        else:
            plt.title(f'Scatter Plot: {feature1} vs {feature2}', fontsize=14)
        
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        filename = "most_similar_features.png" if is_best else f"{feature1.replace(' ', '_').replace('/', '_')}_vs_{feature2.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(f"{self.output_dir}/{filename}")
        plt.close()
        
    def _get_common_data(self, feature1: str, feature2: str):
        mask = self.df[feature1].notna() & self.df[feature2].notna()
        return self.df[mask]  

def main():
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py <dataset.csv>")
        sys.exit(1)
    
    scatter = ScatterPlot(sys.argv[1])
    scatter.find_and_plot_most_similar_features()

if __name__ == "__main__":
    main()