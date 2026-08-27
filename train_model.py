"""
train_model.py
----------------
This script trains a simple Machine Learning model that estimates a
health insurance premium based on: age, sex, bmi, children, smoker, region.

WHY THIS FILE EXISTS
---------------------
The chatbot app (app.py) expects a trained model file called
`health_insurance_model.pkl` to already exist. Since no real trained
model was provided, this script builds one for you so the app can run
end-to-end. If you already have your own trained model (e.g. trained on
the classic "insurance.csv" Kaggle dataset), you can skip this script and
just place your own `health_insurance_model.pkl` in this folder instead
(see README.md for the exact instructions).

This script uses a synthetically generated dataset that mimics realistic
relationships between the inputs and premium cost (higher age, higher
BMI, and smoking status all increase the estimated premium). It is meant
for demonstration / educational purposes only — NOT real insurance data.

Run it with:
    python train_model.py

It will create/overwrite `health_insurance_model.pkl` in this folder.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

RANDOM_SEED = 42
N_SAMPLES = 3000


def generate_synthetic_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """Create a synthetic dataset with realistic-ish relationships."""
    rng = np.random.default_rng(RANDOM_SEED)

    age = rng.integers(18, 65, size=n_samples)
    sex = rng.choice(["male", "female"], size=n_samples)
    bmi = np.round(rng.normal(loc=28, scale=6, size=n_samples).clip(15, 50), 1)
    children = rng.integers(0, 5, size=n_samples)
    smoker = rng.choice(["yes", "no"], size=n_samples, p=[0.2, 0.8])
    region = rng.choice(
        ["northeast", "northwest", "southeast", "southwest"], size=n_samples
    )

    # Base premium formula with noise, in INR, loosely inspired by how
    # age/BMI/smoking tend to drive real-world premiums upward.
    base = 5000
    premium = (
        base
        + age * 250
        + (bmi - 21) * 180
        + children * 800
        + (smoker == "yes") * 18000
        + rng.normal(0, 2500, size=n_samples)
    )
    premium = np.clip(premium, 3000, None).round(0)

    return pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
            "premium": premium,
        }
    )


def build_pipeline() -> Pipeline:
    """Build a preprocessing + model pipeline.

    Categorical columns (sex, smoker, region) are one-hot encoded.
    Numeric columns (age, bmi, children) pass through unchanged.
    """
    categorical_features = ["sex", "smoker", "region"]
    numeric_features = ["age", "bmi", "children"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200, max_depth=8, random_state=RANDOM_SEED
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def main():
    print("Generating synthetic training data...")
    df = generate_synthetic_dataset()

    feature_cols = ["age", "sex", "bmi", "children", "smoker", "region"]
    X = df[feature_cols]
    y = df["premium"]

    print("Training model...")
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    output_path = "health_insurance_model.pkl"
    joblib.dump(pipeline, output_path)
    print(f"Model trained and saved to '{output_path}'")

    # Quick sanity check prediction
    sample = pd.DataFrame(
        [{"age": 35, "sex": "male", "bmi": 27.5, "children": 2, "smoker": "no", "region": "southeast"}]
    )
    pred = pipeline.predict(sample)[0]
    print(f"Sample prediction for a 35-year-old non-smoker: ₹{pred:,.0f}")


if __name__ == "__main__":
    main()
