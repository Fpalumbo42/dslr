import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class ScatterPlot:
    def __init__(self, filepath: str):
        self.df = pd.read_csv(filepath)
        self.houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        self.output_dir = "scatter_plots"
    
    def create_all_scatters(self):
        features = self._get_numeric_features()
        
        for feature in feature:
            self.create_scatter(feature.dropna(), len(feature.dropna()))
            
    def create_scatter(self, x : float, y : float):
        plt.scatter(x, y, edgecolor='black', linewidth=0.5)
        plt.savefig(f"{self.output_dir}/test_individual.png")
        plt.close()
        
            

    def _get_numeric_features(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Index' in numeric_cols:
            numeric_cols.remove('Index')
        return numeric_cols
 
def main():
    scatter = ScatterPlot("datasets/dataset_train.csv")

if __name__ == "__main__":
    main()