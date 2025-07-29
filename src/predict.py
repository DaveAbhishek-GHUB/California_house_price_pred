import pandas as pd
import joblib

def make_prediction(input_data: dict):
    model = joblib.load('saved_models/model.pkl')

    input_df = pd.DataFrame([input_data])

     # One-hot encode 'ocean_proximity' to match training
    input_df = pd.get_dummies(input_df)

    expected_cols = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income',
        'ocean_proximity_INLAND', 'ocean_proximity_ISLAND',
        'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN'
    ]

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_cols]

    # prediction
    prediction = model.predict(input_df)[0]
    return prediction