import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import Utils
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Tuple

class PredictLogisticRegression(): 
    def __init__(self, data_path: str, model_path: str) -> None:
        self.df = pd.read_csv(data_path)

        with open(model_path) as model:
            self.model = json.load(model)
            
        self.weights = {}
        for house in self.model["weights"]:
            self.weights[house] = self.model["weights"][house]
        
        print(json.dumps(self.weights, sort_keys=True, indent=4 ))

        self.features = []
        for feat in self.model["normalization_params"]:
            self.features.append(feat)
        
        self.means = {}
        self.stds = {}
        for feat in self.features:
            self.means[feat] = self.model["normalization_params"][feat]["mean"]
            self.stds[feat] = self.model["normalization_params"][feat]["std"]
        
            
    def normalization(self):
        X = self.df[self.features].copy()
        
        for feat in self.features:
            mean = self.means[feat]
            std = self.stds[feat]
            X[feat] = (X[feat] - mean) / std

        X = X.fillna(0)
        X.insert(0, 'bias', 1)
        print(f"My x {X}")
        
        return X
    
    def get_best_house(self, d: dict):
        max = None
        max_proba = -float('inf')
        
        for key, prob in d.items():
            if prob > max_proba:
                max_proba = prob
                max = key
        
        return max
    
    def save_result(self, pred):
        result = pd.DataFrame({'Hogwarts House': pred})
        result.to_csv('houses.csv', index_label='Index')
        
    def predict(self):
        X = self.normalization()
        
        pred = []
        print(X)
        
        for index, student in X.iterrows():
            proba = {}
            for house in self.weights:
                z = self.weights[house]["bias"] * student['bias']
                for feature in self.features:
                    z += self.weights[house][feature] * student[feature]
                proba[house] = 1 / (1 + np.exp(-z))
            print(json.dumps(proba, indent=4))
            predicted_house = self.get_best_house(proba)
            pred.append(predicted_house)
        
        self.save_result(pred)
        print(pred)

def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py <dataset.csv> <model.json>")
        sys.exit(1)
    predicter = PredictLogisticRegression(sys.argv[1], sys.argv[2])
    predicter.predict()
    
if __name__ == "__main__":
    main()