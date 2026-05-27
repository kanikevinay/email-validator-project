"""Incremental training pipeline for large-scale email classification."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

from .preprocessing import clean_text

DEFAULT_TEXT_COLUMN = "text"
DEFAULT_LABEL_COLUMN = "label"
DEFAULT_CHUNKSIZE = 50_000
DEFAULT_MODEL_FILENAME = "email_classifier_bundle.pkl"
DEFAULT_N_FEATURES = 2 ** 20


def infer_columns(csv_path: Path, text_column: str | None, label_column: str | None) -> tuple[str, str]:
    """Resolve text and label columns, falling back to common defaults."""
    sample = pd.read_csv(csv_path, nrows=5)
    columns = list(sample.columns)

    resolved_text = text_column or (DEFAULT_TEXT_COLUMN if DEFAULT_TEXT_COLUMN in columns else None)
    resolved_label = label_column or (DEFAULT_LABEL_COLUMN if DEFAULT_LABEL_COLUMN in columns else None)

    if resolved_text is None:
        for candidate in ("email", "body", "content", "message", "text"):
            if candidate in columns:
                resolved_text = candidate
                break

    if resolved_label is None:
        for candidate in ("category", "class", "label", "target", "spam_label"):
            if candidate in columns:
                resolved_label = candidate
                break

    if resolved_text is None or resolved_label is None:
        raise ValueError(
            "Could not infer text and label columns. Pass --text-column and --label-column explicitly."
        )

    return resolved_text, resolved_label


def collect_classes(csv_path: Path, label_column: str, chunksize: int) -> list[str]:
    """Stream the dataset once to collect distinct labels without loading all rows."""
    labels: set[str] = set()
    for chunk in pd.read_csv(csv_path, usecols=[label_column], chunksize=chunksize, dtype=str):
        cleaned = chunk[label_column].dropna().astype(str).str.strip()
        labels.update(label for label in cleaned if label)

    if not labels:
        raise ValueError(f"No labels found in column '{label_column}'.")

    return sorted(labels)


def stream_training_rows(
    csv_path: Path,
    text_column: str,
    label_column: str,
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    """Yield cleaned training rows from the source CSV in chunks."""
    for chunk in pd.read_csv(csv_path, usecols=[text_column, label_column], chunksize=chunksize, dtype=str):
        chunk = chunk.dropna(subset=[text_column, label_column]).copy()
        chunk[text_column] = chunk[text_column].astype(str).map(clean_text)
        chunk[label_column] = chunk[label_column].astype(str).str.strip()
        chunk = chunk[(chunk[text_column] != "") & (chunk[label_column] != "")]
        if not chunk.empty:
            yield chunk


def train_model(
    csv_path: str | Path,
    model_dir: str | Path = "models",
    text_column: str | None = None,
    label_column: str | None = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
    n_features: int = DEFAULT_N_FEATURES,
) -> dict:
    """Train an incremental classifier on a large CSV dataset."""
    csv_path = Path(csv_path)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    resolved_text_column, resolved_label_column = infer_columns(csv_path, text_column, label_column)
    classes = collect_classes(csv_path, resolved_label_column, chunksize)

    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        ngram_range=(1, 2),
    )

    classifier = SGDClassifier(loss="log_loss", random_state=42)
    seen_samples = 0

    for chunk in stream_training_rows(csv_path, resolved_text_column, resolved_label_column, chunksize):
        texts = chunk[resolved_text_column].tolist()
        labels = chunk[resolved_label_column].tolist()
        features = vectorizer.transform(texts)

        if seen_samples == 0:
            classifier.partial_fit(features, labels, classes=classes)
        else:
            classifier.partial_fit(features, labels)
        seen_samples += len(chunk)

    if seen_samples == 0:
        raise ValueError("Training data produced zero usable rows after preprocessing.")

    bundle = {
        "classifier": classifier,
        "vectorizer": vectorizer,
        "classes": classes,
        "text_column": resolved_text_column,
        "label_column": resolved_label_column,
        "chunksize": chunksize,
        "n_features": n_features,
        "source_file": str(csv_path),
    }

    model_path = model_dir / DEFAULT_MODEL_FILENAME
    with model_path.open("wb") as handle:
        pickle.dump(bundle, handle)

    metadata = {
        "model_path": str(model_path),
        "classes": classes,
        "text_column": resolved_text_column,
        "label_column": resolved_label_column,
        "chunksize": chunksize,
        "n_features": n_features,
        "rows_seen": seen_samples,
    }
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an email classifier with streaming batches.")
    parser.add_argument("--csv-path", required=True, help="Path to the training CSV file.")
    parser.add_argument("--model-dir", default="models", help="Directory to store the trained bundle.")
    parser.add_argument("--text-column", default=None, help="Name of the email text column.")
    parser.add_argument("--label-column", default=None, help="Name of the label column.")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE, help="Rows per training chunk.")
    parser.add_argument("--n-features", type=int, default=DEFAULT_N_FEATURES, help="HashingVectorizer dimensionality.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    metadata = train_model(
        csv_path=args.csv_path,
        model_dir=args.model_dir,
        text_column=args.text_column,
        label_column=args.label_column,
        chunksize=args.chunksize,
        n_features=args.n_features,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
