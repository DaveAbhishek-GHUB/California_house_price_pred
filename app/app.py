from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "saved_models", "model.pkl")
model = joblib.load(model_path)

# Define expected columns (same as training)
expected_cols = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    'ocean_proximity_INLAND', 'ocean_proximity_ISLAND',
    'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN'
]

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        input_data = {
            'longitude': float(data['longitude']),
            'latitude': float(data['latitude']),
            'housing_median_age': float(data['housing_median_age']),
            'total_rooms': float(data['total_rooms']),
            'total_bedrooms': float(data['total_bedrooms']),
            'population': float(data['population']),
            'households': float(data['households']),
            'median_income': float(data['median_income']),
            'ocean_proximity': data['ocean_proximity']
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)

        # Add missing columns
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder
        input_df = input_df[expected_cols]

        # Predict
        prediction = model.predict(input_df)[0]
        return render_template('form.html', prediction=f"${prediction:,.2f}")

    except Exception as e:
        return render_template('form.html', prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)