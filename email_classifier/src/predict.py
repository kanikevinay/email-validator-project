"""Prediction helpers for the email classification model."""
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

from .preprocessing import clean_text

DEFAULT_MODEL_FILENAME = "email_classifier_bundle.pkl"


@dataclass(slots=True)
class PredictionResult:
    label: str
    confidence: float
    probabilities: dict[str, float]


class EmailClassifierPredictor:
    """Load a trained bundle and run single or batch predictions."""

    def __init__(self, model_path: str | Path = "models") -> None:
        model_path = Path(model_path)
        if model_path.is_dir():
            model_path = model_path / DEFAULT_MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(f"Model bundle not found: {model_path}")

        with model_path.open("rb") as handle:
            bundle: dict[str, Any] = pickle.load(handle)

        self.classifier: SGDClassifier = bundle["classifier"]
        self.vectorizer: HashingVectorizer = bundle["vectorizer"]
        self.classes: list[str] = list(bundle["classes"])
        self.text_column: str = bundle.get("text_column", "text")
        self.label_column: str = bundle.get("label_column", "label")
        self.model_path = model_path

    def _vectorize(self, texts: list[str]):
        cleaned_texts = [clean_text(text) for text in texts]
        return self.vectorizer.transform(cleaned_texts)

    def predict_text(self, text: str) -> PredictionResult:
        features = self._vectorize([text])
        label = self.classifier.predict(features)[0]
        probabilities_array = self.classifier.predict_proba(features)[0]
        probabilities = {class_name: float(prob) for class_name, prob in zip(self.classifier.classes_, probabilities_array)}
        confidence = float(max(probabilities_array))
        return PredictionResult(label=str(label), confidence=confidence, probabilities=probabilities)

    def predict_batch(self, texts: list[str]) -> pd.DataFrame:
        if not texts:
            return pd.DataFrame(columns=["prediction", "confidence"])

        features = self._vectorize(texts)
        predictions = self.classifier.predict(features)
        probabilities = self.classifier.predict_proba(features)
        confidence = probabilities.max(axis=1)

        output = pd.DataFrame(
            {
                "prediction": predictions,
                "confidence": confidence.astype(float),
            }
        )
        return output

    def predict_dataframe(self, frame: pd.DataFrame, text_column: str) -> pd.DataFrame:
        if text_column not in frame.columns:
            raise KeyError(f"Column '{text_column}' not found in uploaded file.")

        result = self.predict_batch(frame[text_column].fillna("").astype(str).tolist())
        combined = frame.copy().reset_index(drop=True)
        combined["prediction"] = result["prediction"]
        combined["confidence"] = result["confidence"]
        return combined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run predictions using a saved email classifier bundle.")
    parser.add_argument("--model-path", default="models", help="Path to the bundle or model directory.")
    parser.add_argument("--text", default=None, help="Single text string to classify.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    predictor = EmailClassifierPredictor(args.model_path)

    if args.text is None:
        raise SystemExit("Provide --text to run a single prediction.")

    result = predictor.predict_text(args.text)
    print(
        {
            "label": result.label,
            "confidence": round(result.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in result.probabilities.items()},
        }
    )


if __name__ == "__main__":
    main()
