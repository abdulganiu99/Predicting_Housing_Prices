import joblib
import pandas as pd
from pathlib import Path

def load_model_bundle(model_path: Path):
    """Loads the deployment bundle (pipeline and model)."""
    print(f"Loading the deployment bundle from {model_path}...")
    bundle = joblib.load(model_path)
    print("✅ Bundle loaded successfully.")
    return bundle

def make_predictions(data: pd.DataFrame, bundle: dict) -> list:
    """
    Uses the loaded pipeline and model to make predictions on new data.
    """
    pipeline = bundle["pipeline"]
    model = bundle["model"]

    print("\nMaking predictions on new data...")
    # Use the loaded pipeline to transform the new data
    # IMPORTANT: Use .transform() here, NOT .fit_transform()
    prepared_data = pipeline.transform(data)

    # Use the loaded model to make predictions
    predictions = model.predict(prepared_data)
    return predictions

if __name__ == "__main__":
    # This block runs when the script is executed directly
    
    # Define the path to the model bundle
    MODEL_BUNDLE_PATH = Path(__file__).parent.parent / "models/housing_model_bundle.pkl"

    # --- 1. Load the Deployment Bundle ---
    deployment_bundle = load_model_bundle(MODEL_BUNDLE_PATH)

    # --- 2. Prepare Some New, Unseen Data ---
    # This should be raw data, just like the original CSV
    new_data = pd.DataFrame([
        {
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY"
        },
        {
            "longitude": -118.3,
            "latitude": 34.2,
            "housing_median_age": 25.0,
            "total_rooms": 3000.0,
            "total_bedrooms": 500.0,
            "population": 1500.0,
            "households": 450.0,
            "median_income": 4.5,
            "ocean_proximity": "<1H OCEAN"
        }
    ])

    # --- 3. Make Predictions ---
    final_predictions = make_predictions(new_data, deployment_bundle)

    # --- 4. Display Results ---
    print("\n--- Prediction Results ---")
    for i, prediction in enumerate(final_predictions):
        print(f"Prediction for House #{i+1}: ${prediction:,.2f}")
