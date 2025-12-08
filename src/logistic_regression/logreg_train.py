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
        data = preprocess_data(self.df)
        print(f"Data shape after preprocessing: {data.shape}")
        
        # Placeholder for training logic
        print("Training logistic regression model...")
        # Here you would implement the actual training logic
        
        print("Training completed.")


def preprocess_data(df):
    print("Preprocessing data...")
    df = df[SELECTED_FEATURES]
    df = df.dropna()
    for feature in SELECTED_FEATURES:
        mean = Utils.get_mean(df[feature])
        std = Utils.get_std(df[feature])
        df[feature] = (df[feature] - mean) / std
    df.insert(0, 'bias', 1)
    return df

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset.csv>")
        sys.exit(1)
    trainer = TrainLogisticRegression(sys.argv[1])
    trainer.train()

if __name__ == "__main__":
    main()