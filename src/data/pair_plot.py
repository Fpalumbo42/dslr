from utils import Utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class PairPlot:
    def __init__(self, filepath: str):
        self.df = pd.read_csv(filepath)
        self.houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        self.output_dir = "pair_plots"
        Utils.create_output_directory(self.output_dir)
        self.plot_colors = {
            'Gryffindor': 'red',
            'Hufflepuff': 'yellow',
            'Ravenclaw': 'blue',
            'Slytherin': 'green'
        }
    
    def create_pair_plot(self):
        features = Utils.get_numeric_features(self.df)
        n_features = len(features)
        
        fig, axes = plt.subplots(n_features, n_features, figsize=(20, 20))
        fig.suptitle('Pair Plot - All Hogwarts Subjects', fontsize=16, fontweight='bold')
        
        for i in range(n_features):
            for j in range(n_features):
                ax = axes[i, j]
                
                if i == j:
                    self._create_diagonal_histogram(ax, features[i])
                else:
                    self._create_scatter(ax, features[j], features[i])
                
                if i == n_features - 1:
                    ax.set_xlabel(features[j], fontsize=8)
                else:
                    ax.set_xticklabels([])
                
                if j == 0:
                    ax.set_ylabel(features[i], fontsize=8)
                else:
                    ax.set_yticklabels([])
                
                ax.tick_params(labelsize=6)
        
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=self.plot_colors[house], 
                             markersize=8, label=house) 
                  for house in self.houses]
        fig.legend(handles=handles, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pair_plot_matrix.png", dpi=100)
        plt.close()
        
        print(f"Pair plot saved to {self.output_dir}/pair_plot_matrix.png")
    
    def _create_diagonal_histogram(self, ax, feature: str):
        common_data = self.df[self.df[feature].notna()]
        
        if len(common_data) == 0:
            return
        
        all_values = common_data[feature].tolist()
        if not all_values:
            return
        
        min_val = Utils.get_min(all_values)
        max_val = Utils.get_max(all_values)
        n_bins = 20
        bin_width = (max_val - min_val) / n_bins
        bins = [min_val + i * bin_width for i in range(n_bins + 1)]
        
        for house in self.houses:
            house_data = common_data[common_data['Hogwarts House'] == house]
            
            if len(house_data) > 0:
                values = house_data[feature].tolist()
                ax.hist(values, bins=bins, alpha=0.6, 
                       color=self.plot_colors[house], 
                       edgecolor='black', linewidth=0.3)
        
        ax.grid(True, alpha=0.3)
    
    def _create_scatter(self, ax, feature_x: str, feature_y: str):
        common_data = self._get_common_data(feature_x, feature_y)
        
        if len(common_data) == 0:
            return
        
        for house in self.houses:
            house_data = common_data[common_data['Hogwarts House'] == house]
            
            if len(house_data) > 0:
                x = house_data[feature_x]
                y = house_data[feature_y]
                
                ax.scatter(x, y, c=self.plot_colors[house], 
                          alpha=0.6, edgecolor='black', 
                          linewidth=0.3, s=10)
        
        ax.grid(True, alpha=0.3)
    
    def _get_common_data(self, feature1: str, feature2: str):
        mask = self.df[feature1].notna() & self.df[feature2].notna()
        return self.df[mask]

def main():
    pair_plot = PairPlot("datasets/dataset_train.csv")
    pair_plot.create_pair_plot()

if __name__ == "__main__":
    main()
