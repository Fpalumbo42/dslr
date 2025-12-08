import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import Utils
import pandas as pd
import numpy as np
import os
import json
from typing import List, Dict, Tuple

SELECTED_FEATURES = ['Astronomy', 'Defense Against the Dark Arts', 'Herbology', 'Ancient Runes']

class TrainLogisticRegression():
    
    def __init__(self, filepath: str) -> None:
        self.df = pd.read_csv(filepath)
        
    def train(self) -> None:
        print("Starting training process...")
        X, y, normalization_params = preprocess_data(self.df)
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))
    
    def cost_function(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        m = len(y)
        z = X @ theta
        h = self.sigmoid(z)

        epsilon = 1e-7
        h = np.clip(h, epsilon, 1 - epsilon)

        # J(θ) = -(1/m) * Σ[y*log(h) + (1-y)*log(1-h)]
        cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

        return cost

def preprocess_data(df):
    X = df[SELECTED_FEATURES].copy()
    print(X)
    y = df['Hogwarts House'].copy()
    print(y)
    
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    normalization_params = {}
    for feature in SELECTED_FEATURES:
        mean = Utils.get_mean(X[feature])
        std = Utils.get_std(X[feature])
        normalization_params[feature] = {'mean': mean, 'std': std}
        X[feature] = (X[feature] - mean) / std
    
    X.insert(0, 'bias', 1)
    
    return X, y, normalization_params

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset.csv>")
        sys.exit(1)
    trainer = TrainLogisticRegression(sys.argv[1])
    trainer.train()

if __name__ == "__main__":
    main()