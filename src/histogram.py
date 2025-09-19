from utils import Utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class Histogram:

    def __init__(self, filepath: str):
        self.df = pd.read_csv(filepath)
        self.houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        self.output_dir = "histograms"
        Utils.create_output_directory(self.output_dir)
        self.plot_colors = {
            'Gryffindor': 'red',
            'Hufflepuff': 'yellow',
            'Ravenclaw': 'blue',
            'Slytherin': 'green'
        }
        
    def create_all_histogram(self):
        features_list = Utils.get_numeric_features(self.df)
        
        for feature in features_list:
            self._create_histogram(feature)
        
        self._create_overview_histogram(features_list)
    
    def _create_overview_histogram(self, features_list):
        n_features = len(features_list)
        
        cols = 4
        rows = (n_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        fig.suptitle('All Hogwarts Subjects - Histogram Comparison', fontsize=16, fontweight='bold')
        
        if rows == 1:
            axes = [axes] if cols == 1 else axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(features_list):
            ax = axes[i]
            bins = self._create_bins(feature)
            
            for house in self.houses:
                house_data = self.get_house_data(house, feature)
                ax.hist(house_data, bins=bins, alpha=0.6, label=house, 
                       color=self.plot_colors[house], edgecolor='black', linewidth=0.3)

            ax.set_title(feature, fontsize=10, fontweight='bold')
            ax.set_xlabel('Score', fontsize=8)
            ax.set_ylabel('Students', fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/ALL_SUBJECTS_OVERVIEW.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_histogram(self, subject: str):
        bins = self._create_bins(subject)
        
        plt.figure(figsize=(12, 8))

        for house in self.houses:
            house_data = self.get_house_data(house, subject)
            plt.hist(house_data, bins=bins, alpha=0.6, label=house, 
                    color=self.plot_colors[house], edgecolor='black', linewidth=0.5)

        plt.title(f"Histogram of {subject} - All Houses")
        plt.xlabel(subject)
        plt.ylabel("Number of students")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        clean_name = subject.replace(' ', '_').replace('/', '_')
        plt.savefig(f"{self.output_dir}/{clean_name}_individual.png")
        plt.close()

    def get_house_data(self, house: str, subject: str):
        house_mask = self.df['Hogwarts House'] == house
        return self.df[house_mask][subject].dropna()
    
    def get_all_houses_data(self, subject: str):
        all_data = []
        for house in self.houses:
            house_data = self.get_house_data(house, subject)
            all_data.extend(house_data.tolist())
        return all_data
        
    def _create_bins(self, subject: str):
        all_values = self.get_all_houses_data(subject)
        
        if not all_values:
            return []
            
        min_val = Utils.get_min(all_values)
        max_val = Utils.get_max(all_values)
        n_bins = 30
        
        bin_width = (max_val - min_val) / n_bins
        bins = [min_val + i * bin_width for i in range(n_bins + 1)]
        
        return bins

def main():
    histo = Histogram("datasets/dataset_train.csv")
    histo.create_all_histogram()

if __name__ == "__main__":
    main()