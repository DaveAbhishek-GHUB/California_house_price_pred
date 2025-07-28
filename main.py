from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train import train_model

if __name__ == "__main__":
    path = "data/raw/housing.csv"
    df = load_data(path)
    
    if df is not None:
        X_train, X_test, y_train, y_test = preprocess_data(df)

         # Train the model (try 'linear' or 'random_forest')
        model = train_model(X_train, X_test, y_train, y_test, model_type="random_forest")