from utils import Utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class ScatterPlot:
    def __init__(self, filepath: str):
        self.df = pd.read_csv(filepath)
        self.houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        self.output_dir = "scatter_plots"
        Utils.create_output_directory(self.output_dir)
    
    def create_all_scatters(self):
        features = self._get_numeric_features()
        
        for feature in features:
            self.create_scatter(self.df[feature].dropna(), len(self.df[feature].dropna()), feature)
            
    def create_scatter(self, x : float, y : float, feature : str):
        plt.scatter(x, y, edgecolor='black', linewidth=0.5)
        
        clean_name = feature.replace(' ', '_').replace('/', '_')
        plt.savefig(f"{self.output_dir}/{clean_name}.png")
        plt.close()     

    def _get_numeric_features(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Index' in numeric_cols:
            numeric_cols.remove('Index')
        return numeric_cols
 
def main():
    scatter = ScatterPlot("datasets/dataset_train.csv")
    scatter.create_all_scatters()

if __name__ == "__main__":
    main()