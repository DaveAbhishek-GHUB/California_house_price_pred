from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train import train_model
from src.predict import make_prediction

if __name__ == "__main__":
    path = "data/raw/housing.csv"
    df = load_data(path)
    
    if df is not None:
        X_train, X_test, y_train, y_test = preprocess_data(df)

         # Train the model (try 'linear' or 'random_forest')
        model = train_model(X_train, X_test, y_train, y_test, model_type="random_forest")

        # # Example input
        # example_input = {
        #     'longitude': -122.23,
        #     'latitude': 37.88,
        #     'housing_median_age': 41.0,
        #     'total_rooms': 880.0,
        #     'total_bedrooms': 129.0,
        #     'population': 322.0,
        #     'households': 126.0,
        #     'median_income': 8.3252,
        #     'ocean_proximity': 'NEAR BAY'
        # }

        # predicted_price = make_prediction(example_input)
        # print(f"Predicted House Price: {predicted_price:.2f}")