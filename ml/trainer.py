"""
ml/trainer.py - ML Model Training Pipeline
===========================================

This module implements a minimal model training pipeline for admin-led
retraining of disease risk models from uploaded datasets.

It is intentionally lightweight for Phase 3 and supports:
- Discovering CSV/XLSX dataset files in the configured dataset folder.
- Extracting questionnaire-like features using FeaturePipeline.
- Training one binary classifier per disease where the target column exists.
- Saving trained model artifacts to the configured MODELS_FOLDER.
- Recording training metadata in the ModelLog table.
"""

import os
import time
import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from extensions import db
from ml.feature_pipeline import FeaturePipeline
from models.model_log import ModelLog


class ModelTrainer:
    """Minimal training pipeline for admin-triggered model retraining."""

    DISEASES = [
        "diabetes", "heart_disease", "stroke",
        "hypertension", "obesity", "kidney_disease",
        "fatty_liver", "depression", "sleep_disorder", "thyroid",
    ]

    TARGET_COLUMN_ALIASES = {
        "obesity": "obesity",
        "obesity_type_i": "obesity",
        "obesity_type_ii": "obesity",
        "obesity_type_iii": "obesity",
        "normal_weight": "obesity",
        "overweight_level_i": "obesity",
        "overweight_level_ii": "obesity",
        "overweight_level_iii": "obesity",
        "sleep disorder": "sleep_disorder",
        "sleep_disorder": "sleep_disorder",
        "sleep_apnea": "sleep_disorder",
        "insomnia": "sleep_disorder",
    }

    POSITIVE_LABELS = {"1", "yes", "true", "positive", "y", "t", "high"}
    NEGATIVE_LABELS = {"0", "no", "false", "negative", "n", "f", "low"}

    def __init__(self, app):
        self.app = app
        self.dataset_folder = app.config["DATASET_FOLDER"]
        self.models_folder = app.config["MODELS_FOLDER"]
        os.makedirs(self.models_folder, exist_ok=True)

    def train_from_datasets(self) -> dict:
        """Train disease models from uploaded dataset files."""
        dataset_files = self._discover_dataset_files()
        if not dataset_files:
            return {
                "trained_models": [],
                "skipped_models": [],
                "message": "No dataset files found in the datasets folder.",
            }

        merged_df = self._load_dataset_files(dataset_files)
        if merged_df is None or merged_df.empty:
            return {
                "trained_models": [],
                "skipped_models": [],
                "message": "No valid dataset rows were loaded from the uploaded files.",
            }

        valid_targets = []
        seen_diseases = set()
        for column in merged_df.columns:
            canonical = self._map_target_column(column)
            if canonical and canonical not in seen_diseases:
                valid_targets.append((column, canonical))
                seen_diseases.add(canonical)

        if not valid_targets:
            return {
                "trained_models": [],
                "skipped_models": [],
                "message": (
                    "Dataset files were loaded, but no disease target columns were found. "
                    "Please include one or more of: " + ", ".join(self.DISEASES)
                ),
            }

        trained_models = []
        skipped_models = []

        for raw_column, disease in valid_targets:
            target_series = merged_df[raw_column].dropna()
            if target_series.empty:
                skipped_models.append({
                    "disease": disease,
                    "reason": f"Target column '{raw_column}' contains no values.",
                })
                continue

            labeled_rows = merged_df.loc[target_series.index]
            feature_dicts = self._extract_feature_dicts(labeled_rows)
            if not feature_dicts:
                skipped_models.append({
                    "disease": disease,
                    "reason": f"No valid feature rows could be built for target '{raw_column}'.",
                })
                continue

            y = self._normalize_target(target_series, disease)
            if y is None:
                skipped_models.append({
                    "disease": disease,
                    "reason": f"Could not interpret target values for column '{raw_column}'.",
                })
                continue

            if len(feature_dicts) != len(y):
                paired = list(zip(feature_dicts, y))
                feature_dicts, y = zip(*paired)
                feature_dicts = list(feature_dicts)
                y = np.asarray(y, dtype=np.int8)

            vectorizer = DictVectorizer(sparse=False)
            X = vectorizer.fit_transform(feature_dicts)

            if len(np.unique(y)) != 2:
                skipped_models.append({
                    "disease": disease,
                    "reason": "Target column must contain two classes for binary classification.",
                })
                continue

            if sum(y == 1) < 5 or sum(y == 0) < 5:
                skipped_models.append({
                    "disease": disease,
                    "reason": "Target column needs at least 5 positive and 5 negative examples.",
                })
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.20,
                stratify=y,
                random_state=42,
            )

            classifier = RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
            )
            classifier.fit(X_train, y_train)

            metrics = self._evaluate_model(classifier, X_test, y_test)
            model_filename = self._save_model_artifact(disease, vectorizer, classifier)
            self._archive_previous_model_logs(disease)

            model_log = ModelLog(
                disease=disease,
                algorithm="RandomForest",
                version=f"rf-{int(time.time())}",
                is_active=True,
                model_path=model_filename,
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
                roc_auc=metrics["roc_auc"],
                extra_metrics=json.dumps(metrics.get("extra", {})),
                training_samples=len(y),
                training_duration=metrics["duration"],
                trained_by=current_admin_email(self.app),
            )
            db.session.add(model_log)
            db.session.commit()

            trained_models.append({
                "disease": disease,
                "model_path": model_filename,
                "metrics": metrics,
            })

        message = (
            f"Trained {len(trained_models)} model(s). "
            f"Skipped {len(skipped_models)} model(s)."
        )
        return {
            "trained_models": trained_models,
            "skipped_models": skipped_models,
            "message": message,
        }

    def _discover_dataset_files(self) -> List[str]:
        """Return sorted list of dataset files in the dataset folder."""
        if not os.path.isdir(self.dataset_folder):
            return []
        files = []
        for filename in sorted(os.listdir(self.dataset_folder)):
            path = os.path.join(self.dataset_folder, filename)
            if os.path.isfile(path) and filename.lower().endswith((".csv", ".xlsx", ".xls")):
                files.append(path)
        return files

    def _load_dataset_files(self, file_paths: List[str]) -> Optional[pd.DataFrame]:
        """Load CSV/XLSX dataset files and concatenate them into one DataFrame."""
        dfs = []
        for path in file_paths:
            try:
                if path.lower().endswith(".csv"):
                    df = pd.read_csv(path)
                else:
                    df = pd.read_excel(path)
            except Exception:
                continue
            if df is not None and not df.empty:
                dfs.append(df)

        if not dfs:
            return None

        merged = pd.concat(dfs, ignore_index=True)
        return merged

    def _map_target_column(self, column_name: str) -> Optional[str]:
        normalized = str(column_name).strip().lower()
        normalized = normalized.replace(" ", "_").replace("-", "_")
        if normalized in self.TARGET_COLUMN_ALIASES:
            return self.TARGET_COLUMN_ALIASES[normalized]
        if normalized in self.DISEASES:
            return normalized
        return None

    def _extract_feature_dicts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert raw dataset rows into feature dictionaries for the ML pipeline."""
        records = []
        for _, row in df.iterrows():
            feature_dict = FeaturePipeline.extract_features(row.to_dict())
            cleaned = {k: v for k, v in feature_dict.items() if v is not None}
            if cleaned:
                records.append(cleaned)
        return records

    def _normalize_target(self, series: pd.Series, disease: Optional[str] = None) -> Optional[np.ndarray]:
        """Convert a target column to a binary numpy array if possible."""
        if series.empty:
            return None

        values = series.dropna().astype(str).str.strip()
        if values.empty:
            return None

        lower_values = values.str.lower()
        uniques = set(lower_values.unique())

        if uniques <= self.POSITIVE_LABELS | self.NEGATIVE_LABELS:
            return lower_values.map(lambda v: 1 if v in self.POSITIVE_LABELS else 0).to_numpy(dtype=np.int8)

        try:
            numeric = pd.to_numeric(series, errors="coerce")
        except Exception:
            numeric = pd.Series(dtype=float)
        numeric = numeric.dropna()
        if not numeric.empty:
            unique_numeric = np.unique(numeric)
            if set(unique_numeric) <= {0, 1}:
                return numeric.astype(np.int8).to_numpy()
            if len(unique_numeric) == 2:
                mapping = {unique_numeric[0]: 0, unique_numeric[1]: 1}
                return numeric.map(mapping).astype(np.int8).to_numpy()

        if len(uniques) == 2:
            categories = list(uniques)
            mapping = {}
            for val in categories:
                if val in self.POSITIVE_LABELS:
                    mapping[val] = 1
                elif val in self.NEGATIVE_LABELS:
                    mapping[val] = 0
            if len(mapping) == 2:
                return lower_values.map(lambda v: mapping[v]).to_numpy(dtype=np.int8)

            for val in categories:
                if any(token in val for token in ("none", "normal", "no", "negative", "healthy")):
                    mapping[val] = 0
                else:
                    mapping[val] = 1
            if len(mapping) == 2:
                return lower_values.map(lambda v: mapping[v]).to_numpy(dtype=np.int8)

        if disease == "obesity":
            positive = lower_values.str.contains("obese|overweight|type")
            negative = lower_values.str.contains("normal|none|healthy")
            if positive.any() and negative.any():
                return positive.astype(int).to_numpy(dtype=np.int8)

        if disease == "sleep_disorder":
            positive = lower_values.str.contains("apnea|insomnia|disorder|sleep")
            negative = lower_values.str.contains("none|normal|healthy")
            if positive.any() and negative.any():
                return positive.astype(int).to_numpy(dtype=np.int8)

        return None

    def _evaluate_model(self, classifier, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Compute standard classification metrics for a binary disease model."""
        import time
        start = time.time()
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = None
        if hasattr(classifier, "predict_proba") and len(np.unique(y_test)) == 2:
            try:
                y_score = classifier.predict_proba(X_test)[:, 1]
                roc_auc = float(roc_auc_score(y_test, y_score))
            except Exception:
                roc_auc = None

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
            "duration": time.time() - start,
            "extra": {},
        }

    def _save_model_artifact(self, disease: str, vectorizer: DictVectorizer, classifier) -> str:
        """Save the trained model and vectorizer to disk and return the filename."""
        timestamp = int(time.time())
        filename = f"{disease}_rf_{timestamp}.joblib"
        path = os.path.join(self.models_folder, filename)
        joblib.dump({"vectorizer": vectorizer, "classifier": classifier}, path)
        return filename

    def _archive_previous_model_logs(self, disease: str) -> None:
        """Mark previous active logs for the same disease as inactive."""
        ModelLog.query.filter_by(disease=disease, is_active=True).update({"is_active": False})
        db.session.commit()


def current_admin_email(app) -> Optional[str]:
    """Return the current user email if it is an admin session, otherwise None."""
    try:
        from flask_login import current_user
        if current_user.is_authenticated and getattr(current_user, "is_admin", False):
            return getattr(current_user, "email", None)
    except Exception:
        pass
    return None
