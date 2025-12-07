# c:\Users\DELL\Downloads\ML_Projects\Predicting_Housing_Prices\api_fastapi.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
import sys

# Add the project root to the Python path to allow imports from the 'scripts' folder
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Initialize the FastAPI app
app = FastAPI(title="Housing Price Prediction API")

# --- Define the request body structure using Pydantic ---
class HousingData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

# --- Load the model bundle at startup ---
# This ensures the model is loaded only once
# We need to import the custom transformer for joblib to load the pipeline
from scripts.custom_transformers import CombinedAttributesAdder

MODEL_BUNDLE_PATH = PROJECT_ROOT / "models/housing_model_bundle.pkl"
bundle = joblib.load(MODEL_BUNDLE_PATH)
print(f"✅ Model bundle loaded from {MODEL_BUNDLE_PATH}")

# --- Define the prediction endpoint ---
@app.post("/predict")
def predict(data: HousingData):
    """
    Takes housing data as input and returns a price prediction.
    """
    # Convert the input data to a pandas DataFrame
    df = pd.DataFrame([data.dict()])

    # Use the pipeline and model from the loaded bundle
    pipeline = bundle["pipeline"]
    model = bundle["model"]

    # Make prediction
    prepared_data = pipeline.transform(df)
    prediction = model.predict(prepared_data)[0]

    return {"predicted_price": prediction}

@app.get("/")
def root():
    return {"message": "Welcome to the Housing Price Prediction API. Go to /docs for details."}
