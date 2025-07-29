from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_model(X_train, X_test, y_train, y_test, model_type='linear'):
    if model_type == 'linear':
        model = LinearRegression()
        print("Using Linear Regression model.")

    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42)
        print("Using random forest model.")

    # train the model
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    # model Evalutioln
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"Model trained.")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # Save model
    joblib.dump(model, "saved_models/model.pkl")
    print("Model saved to saved_models/model.pkl")

    return model
