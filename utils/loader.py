import pandas as pd

def load_pca_data():
    return pd.read_csv("E:/tokyo/work/recipe-cluster-app/recipe-cluster-app/data/pca_result.csv")

import joblib

def load_processed_data():
    return pd.read_csv("E:/tokyo/work/recipe-cluster-app/recipe-cluster-app/data/processed_recipes.csv")

def load_tfidf_vectorizer(path: str):
    return joblib.load(path)
