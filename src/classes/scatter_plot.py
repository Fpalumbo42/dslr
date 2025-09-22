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
        self.plot_colors = {
            'Gryffindor': 'red',
            'Hufflepuff': 'yellow',
            'Ravenclaw': 'blue',
            'Slytherin': 'green'
        }
    
    def create_all_scatters(self):
        features = Utils.get_numeric_features(self.df)
        
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                feature1 = features[i]
                feature2 = features[j]
                self.create_scatter(feature1, feature2)
            
    def create_scatter(self, feature1: str, feature2: str):
        common_data = self._get_common_data(feature1, feature2)
        
        if len(common_data) == 0:
            return
        
        plt.figure(figsize=(10, 8))
        
        for house in self.houses:
            house_data = common_data[common_data['Hogwarts House'] == house]
            
            if len(house_data) > 0:
                x = house_data[feature1]
                y = house_data[feature2]
                
                plt.scatter(x, y, c=self.plot_colors[house], label=house,alpha=0.6,edgecolor='black', linewidth=0.3,s=30)
        
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f'Scatter Plot: {feature1} vs {feature2}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.output_dir}/{feature1.replace(' ', '_').replace('/', '_')}_vs_{feature2.replace(' ', '_').replace('/', '_')}.png")
        plt.close()
        
    def _get_common_data(self, feature1: str, feature2: str):
        mask = self.df[feature1].notna() & self.df[feature2].notna()
        return self.df[mask]  

def main():
    scatter = ScatterPlot("datasets/dataset_train.csv")
    scatter.create_all_scatters()

if __name__ == "__main__":
    main()