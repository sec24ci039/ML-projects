"""Train and use a simple job recommendation model based on skills and education."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train(data_path: str = "data/sample_job_data.csv", model_path: str = "models/job_recommendation_model.pkl") -> None:
    """Train a logistic regression model and save it to disk."""
    df = pd.read_csv(data_path)
    X = df.drop("job_role", axis=1)
    y = df["job_role"]

    categorical = ["education_level"]
    numeric = [c for c in X.columns if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    # Use a relatively large test size to ensure each class is represented
    # when performing a stratified split on this small demonstration dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def predict(model_path: str, profile: dict) -> str:
    """Load a trained model and predict job role for a single profile."""
    model = joblib.load(model_path)
    df = pd.DataFrame([profile])
    return model.predict(df)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Job role recommender")
    parser.add_argument("action", choices=["train", "predict"], help="train the model or predict a job role")
    parser.add_argument("--model", default="models/job_recommendation_model.pkl", help="Path to model file")
    parser.add_argument("--data", default="data/sample_job_data.csv", help="Training data path")
    parser.add_argument("--education", help="Education level for prediction")
    parser.add_argument("--python", type=int, default=0, help="Python skill (1 or 0)")
    parser.add_argument("--sql", type=int, default=0, help="SQL skill (1 or 0)")
    parser.add_argument("--statistics", type=int, default=0, help="Statistics skill (1 or 0)")
    parser.add_argument("--java", type=int, default=0, help="Java skill (1 or 0)")
    parser.add_argument("--cloud", type=int, default=0, help="Cloud skill (1 or 0)")
    args = parser.parse_args()

    if args.action == "train":
        train(args.data, args.model)
    else:
        profile = {
            "education_level": args.education,
            "python": args.python,
            "sql": args.sql,
            "statistics": args.statistics,
            "java": args.java,
            "cloud": args.cloud,
        }
        print(predict(args.model, profile))
