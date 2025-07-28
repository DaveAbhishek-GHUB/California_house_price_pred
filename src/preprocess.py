import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def preprocess_data(df: pd.DataFrame):
    df = df.dropna()

    # Separate features and target
    X = df.drop('median_house_value', axis = 1)
    y = df['median_house_value']

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=['ocean_proximity'], drop_first=True)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=20, random_state=42)

    print("Preprocessing done.")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test
