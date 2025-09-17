import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class Histogram:

    def __init__(self, filepath: str):
        self.df = pd.read_csv(filepath)
        self.houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        self.output_dir = "histograms"
        self._create_output_directory()
        
    def create_all_histogram(self):
        features_list = self._get_numeric_features()
        for feature in features_list:
            self._create_histogram(feature)
    
    def _create_histogram(self, subject: str):
        bins = self._create_bins(subject)
        print(bins)
        
        plt.figure(figsize=(12, 8))
        
        colors = {'Gryffindor' : 'red',
                  'Hufflepuff' : 'yellow',
                  'Ravenclaw' : 'blue',
                  'Slytherin' : 'green'
                  }
        
        for i, house in enumerate(self.houses):
            house_data = self.get_house_data(house, subject)
            print(f"{house} - {subject}: {len(house_data)} students")
            print(f"Sample values: {house_data.head().tolist()}")
            
            plt.hist(house_data, bins=bins, alpha=0.6, label=house, 
                    color=colors[house], edgecolor='black', linewidth=0.5)
        
        plt.title(f"Histogram of {subject} - All Houses")
        plt.xlabel(subject)
        plt.ylabel("Number of students")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        plt.savefig(f"{self.output_dir}/{subject}.png")
    
    def _create_output_directory(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory: {self.output_dir}")
        else:
            print(f"Directory {self.output_dir} already exists")


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
            
        min_val = self._get_min(all_values)
        max_val = self._get_max(all_values)
        n_bins = 30
        
        print(f"{subject} - Range: [{min_val:.2f}, {max_val:.2f}]")
        print(f"Total students: {len(all_values)}")
        
        bin_width = (max_val - min_val) / n_bins
        bins = [min_val + i * bin_width for i in range(n_bins + 1)]
        
        return bins
    
    def _get_min(self, values: list) -> float:
        if len(values) == 0:
            return float('nan')
        
        tmp = values[0]
        for value in values:
            if value < tmp:
                tmp = value
        return float(tmp)
    
    def _get_max(self, values: list) -> float:
        if len(values) == 0:
            return float('nan')
        
        tmp = values[0]
        for value in values:
            if value > tmp:
                tmp = value
        return float(tmp)
    
    def _get_numeric_features(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('Index')
        return numeric_cols

def main():
    histo = Histogram("datasets/dataset_train.csv")
    histo.create_all_histogram()

if __name__ == "__main__":
    main()