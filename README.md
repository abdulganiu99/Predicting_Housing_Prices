## 🧠 Project Summary (Prelaunch Insights)

This project explored building a machine learning system to estimate housing prices in California.  
Beyond implementing the model, the focus was on understanding the data, identifying what mattered most,  
and evaluating how well the system generalizes to unseen districts.

### Key Findings
- **Median income is the strongest predictor of housing prices.**
- Houses near the ocean or bay consistently command higher values.
- Engineered features such as *bedrooms_per_room* and *population_per_household* improved model performance.
- Decision Trees severely overfit; Random Forests struck the best balance between bias and variance.

### Model Performance
- Final Test RMSE: **$47,8xx**  
- Baseline Median Predictor: significantly worse  
- Performance is comparable to domain experts but not superior  

## ⚠️ Assumptions & Limitations

- The dataset represents **block-group aggregates**, not individual property-level prices.
- Income levels were **capped and scaled**, which may reduce signal in extreme cases.
- Model assumes **static market conditions**; no temporal trends included.
- Spatial autocorrelation is not modeled — neighborhoods influence each other, but the model treats them as independent.
- Random Forest does not extrapolate beyond seen values (cannot predict above \$500,001 due to dataset cap).

## ✅ What Worked
- Stratified sampling prevented income-skewed splits.
- Custom transformer improved consistency & reproducibility.
- Engineered features added meaningful predictive power.
- Two-stage hyperparameter tuning efficiently found a strong model.

## ❌ What Didn’t Work
- Decision Tree regressor overfit dramatically.
- Linear Regression underfit — unable to capture non-linear relationships.
- Some features had long tails and required scaling to stabilize training.

## 📘 Lessons Learned
- More data (especially on amenities, crime, transit) would significantly improve accuracy.
- Model design must consider both **domain knowledge** and **statistical patterns**.
- Pipelines + transformers = essential for reproducible, deployable ML systems.
- Hyperparameter search should start broad (RandomizedSearchCV) then refine (GridSearchCV).

## 🚀 Launch Decision

Although the model does not outperform human experts, it provides consistent, fast, and scalable estimates.  
It can be used to:

- pre-screen candidate districts  
- assist analysts by reducing manual workload  
- support dashboards and automated reporting tools

This system would free experts to focus on nuanced pricing decisions that require context the dataset does not capture.

## 📊 Geographical Distribution of Housing Prices
![Geographical Scatter Plot](visuals/scatter_geo.png)

## 🔥 Feature Importance
![Feature Importance](visuals/feature_importance.png)

## 📈 Predicted vs Actual Values
![Test vs Predicted](visuals/test_vs_pred.png)

## 🔍 Correlation Matrix
![Correlation Matrix](visuals/correlation_matrix.png)

**Interpretation:**  
Higher median income strongly correlates with higher housing prices.  
Geographical clustering reveals coastal regions have higher valuations.


# 📄 Model Card: California Housing Price Estimator

**Model Type:** RandomForestRegressor  
**Version:** 1.0  
**Intended Use:** Estimate median house values at the block-group level  
**Users:** Analysts, researchers, data teams
**Ethical Considerations:**
- Model is not intended to determine mortgage or insurance rates
- Predictions rely on historical census data, subject to bias

---

## ⚙️ Setup and Usage

### 1. Environment Setup

Clone the repository and install the required dependencies.

```bash
# It is recommended to use a virtual environment
conda create --name housing_env python=3.9
conda activate housing_env

pip install -r requirements.txt
```

### 2. Running Predictions

You can run predictions using the command-line script or by starting the API server.

**Command-Line Prediction:**
```bash
# This will run predictions on hard-coded sample data in the script
python scripts/predict.py
```

**API Server:**
```bash
# Start the FastAPI server
uvicorn scripts.api_fastapi:app --reload
```
Once running, the API documentation will be available at `http://127.0.0.1:8000/docs`.
