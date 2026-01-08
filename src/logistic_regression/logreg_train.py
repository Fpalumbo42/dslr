import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import Utils
import pandas as pd
import numpy as np
import os
import json
from typing import List, Dict, Tuple

SELECTED_FEATURES = ['Astronomy', 'Herbology', 'Ancient Runes', "Divination", "Charms", "Flying"]

class TrainLogisticRegression():

    def __init__(self, filepath: str) -> None:
        self.df = pd.read_csv(filepath)

    def train(self) -> None:
        print("Starting training process...")
        X, y, normalization_params = self.preprocess_data(self.df)

        X_array = X.values
        y_array = y.values

        weights = self.train_one_vs_all(X_array, y_array)
        self.save_model(weights, normalization_params)

    def train_one_vs_all(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        weights_dict = {}

        print(f"\n=== Training One-vs-All classifiers ===")
        print(f"Total students: {len(y)}\n")

        for house in houses:
            print(f"\n--- Training classifier for {house} vs All ---")

            y_binary = (y == house).astype(int)

            num_in_house = np.sum(y_binary)
            print(f"Students in {house}: {num_in_house} / {len(y)}")

            theta = self.gradient_descent(X, y_binary)
            weights_dict[house] = theta

        return weights_dict

    def sigmoid(self, scores: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-scores))

    def cost_function(self, features: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
        num_samples = len(labels)

        scores = features @ weights
        predictions = self.sigmoid(scores)

        epsilon = 1e-7
        predictions = np.clip(predictions, epsilon, 1 - epsilon)

        cost = -(1/num_samples) * np.sum(
            labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions)
        )

        return cost

    def gradient_descent(self, X, y_binary, learning_rate=0.1, iterations=100000, tolerance=1e-6):
        m, n = X.shape
        theta = np.zeros(n)
        cost_history = []
        previous_cost = float('inf')

        for i in range(iterations):
            h = self.sigmoid(X @ theta)

            # Compute gradient: ∇J(θ) = (1/m) * X^T @ (h - y)
            gradient = (1/m) * (X.T @ (h - y_binary))

            # Update weights: θ := θ - α * ∇J(θ)
            theta = theta - learning_rate * gradient

            current_cost = self.cost_function(X, y_binary, theta)
            cost_history.append(current_cost)

            # Display progress
            if i % 100 == 0:
                print(f"  Iteration {i}: Cost = {current_cost:.4f}")

            # Early stopping: check if converged
            if abs(previous_cost - current_cost) < tolerance:
                print(f"Converged at iteration {i}")
                break

            previous_cost = current_cost

        return theta
    
    def save_model(self, weights: Dict[str, np.ndarray], normalization_params: Dict[str, Dict[str, float]], model_path: str = "logreg_model.json") -> None:
        
        weights_feat = {}
        for house in weights:
            house_weights = {}
            
            house_weights["bias"] = float(weights[house][0])
            
            for i, feature_name in enumerate(SELECTED_FEATURES):
                house_weights[feature_name] = float(weights[house][i + 1])
            
            weights_feat[house] = house_weights
        
        model_data = {
            'weights': weights_feat,
            'normalization_params': normalization_params
        }
        with open(model_path, 'w') as f:
            json.dump(model_data, f, indent=4)
        print(f"Model saved to {model_path}")

    def preprocess_data(self, df):
        X = df[SELECTED_FEATURES].copy()
        y = df['Hogwarts House'].copy()

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