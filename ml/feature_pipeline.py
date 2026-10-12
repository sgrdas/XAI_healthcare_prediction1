"""
ml/feature_pipeline.py - Feature Extraction & Preprocessing Pipeline
=====================================================================
Transforms questionnaire and daily log data into standardized numerical
feature vectors for ML models, SHAP explainers, and LIME explainers.
"""

from typing import Dict, Any, List, Tuple
import math

# Baseline population expectations E[f(x)] for baseline SHAP calculations
BASE_POPULATION_EXPECTATIONS = {
    "age": 42.0,
    "bmi": 24.5,
    "sleep_hours": 7.2,
    "water_intake_L": 2.2,
    "exercise_duration_min": 25.0,
    "daily_steps": 6500.0,
    "stress_level": 4.5,
    "fruit_veg_servings": 4.0,
    "fast_food_per_week": 2.0,
    "smoking_active": 0.0,
    "alcohol_heavy": 0.0,
    "family_history_score": 0.5,
    "blood_pressure_systolic": 120.0,
    "blood_sugar_fasting": 95.0,
}

FEATURE_NAMES = list(BASE_POPULATION_EXPECTATIONS.keys())


class FeaturePipeline:
    """
    Standardizes raw lifestyle dicts (from Questionnaire or DailyHealthLog)
    into clean numerical feature vectors suitable for ML & XAI explainers.
    """

    @staticmethod
    def _normalize_key(key: Any) -> str:
        raw = str(key).strip().lower()
        raw = raw.replace(" ", "_").replace("-", "_")
        return raw

    @staticmethod
    def _get_raw_value(raw_dict: Dict[str, Any], *keys: str, default=None):
        normalized = {
            FeaturePipeline._normalize_key(key): value
            for key, value in raw_dict.items()
            if key is not None
        }
        for key in keys:
            normalized_key = FeaturePipeline._normalize_key(key)
            value = normalized.get(normalized_key)
            if value not in (None, "", []):
                return value
        return default

    @staticmethod
    def extract_features(raw_dict: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract numerical features from raw dictionary.

        Args:
            raw_dict: Raw input dict from questionnaire or merged log features.

        Returns:
            Dict mapping standardized feature name -> float value.
        """
        features = {}

        # 1. Age
        features["age"] = float(FeaturePipeline._get_raw_value(raw_dict, "age", default=35))

        # 2. BMI
        raw_height = FeaturePipeline._get_raw_value(
            raw_dict,
            "height_cm", "height", "height (cm)",
            default=None,
        )
        raw_weight = FeaturePipeline._get_raw_value(
            raw_dict,
            "weight_kg", "weight", "weight (kg)",
            default=None,
        )
        bmi = FeaturePipeline._get_raw_value(raw_dict, "bmi")
        height_cm = None
        if raw_height is not None:
            try:
                height_value = float(raw_height)
                if height_value > 0 and height_value < 10:
                    height_cm = height_value * 100.0
                else:
                    height_cm = height_value
            except (TypeError, ValueError):
                height_cm = None

        weight_kg = None
        if raw_weight is not None:
            try:
                weight_kg = float(raw_weight)
            except (TypeError, ValueError):
                weight_kg = None

        if bmi is None and height_cm and weight_kg:
            h_m = height_cm / 100.0
            if h_m > 0:
                bmi = weight_kg / (h_m ** 2)

        bmi_value = bmi if bmi is not None else weight_kg if weight_kg is not None else raw_dict.get("weight", 24.5)
        features["bmi"] = float(bmi_value if bmi_value else 24.5)

        # 3. Sleep
        features["sleep_hours"] = float(raw_dict.get("sleep_hours", 7.0))

        # 4. Water
        features["water_intake_L"] = float(
            raw_dict.get("water_intake_L", raw_dict.get("water_intake", 2.0))
        )

        # 5. Exercise
        ex_min = FeaturePipeline._get_raw_value(
            raw_dict,
            "exercise_duration_min", "exercise_minutes",
            default=None,
        )
        if ex_min is None:
            freq = FeaturePipeline._get_raw_value(
                raw_dict,
                "exercise_frequency",
                default="Never",
            )
            if isinstance(freq, str):
                freq = freq.strip()
            freq_map = {
                "Never": 0.0,
                "1-2x per week": 15.0,
                "3-4x per week": 40.0,
                "5+ times per week": 60.0,
            }
            ex_min = freq_map.get(freq, 20.0)
        features["exercise_duration_min"] = float(ex_min)

        # 6. Steps
        features["daily_steps"] = float(raw_dict.get("daily_steps", raw_dict.get("steps", 5000)))

        # 7. Stress Level
        features["stress_level"] = float(
            FeaturePipeline._get_raw_value(raw_dict, "stress_level", default=5)
        )

        # 8. Fruits & Vegetables
        fv = FeaturePipeline._get_raw_value(raw_dict, "fruit_veg_servings", "fruits", "vegetables", default=None)
        if fv is None:
            fv = 0
        features["fruit_veg_servings"] = float(fv)

        # 9. Fast / Junk Food
        ff = FeaturePipeline._get_raw_value(raw_dict, "fast_food_per_week", "faf", default=None)
        if ff is None:
            ff = 7.0 if FeaturePipeline._get_raw_value(raw_dict, "junk_food") else 1.0
        features["fast_food_per_week"] = float(ff)

        # 10. Smoking status
        smk = FeaturePipeline._get_raw_value(raw_dict, "smoking_status", "smoke", "smoking")
        if smk is None:
            smk_val = 1.0 if FeaturePipeline._get_raw_value(raw_dict, "smoking") else 0.0
        else:
            if isinstance(smk, str):
                smk_val = 1.0 if smk.strip().lower() in ("yes", "y", "current", "smoker", "smoke") else 0.0
            else:
                smk_val = 1.0 if bool(smk) else 0.0
        features["smoking_active"] = float(smk_val)

        # 11. Alcohol
        alc = raw_dict.get("alcohol_intake")
        if alc is None:
            alc_val = 1.0 if raw_dict.get("alcohol") else 0.0
        else:
            alc_val = 1.0 if alc in ("Heavy", "Moderate") else 0.0
        features["alcohol_heavy"] = float(alc_val)

        # 12. Family History Score
        fam_history = FeaturePipeline._get_raw_value(raw_dict, "family_history", default=None)
        fam_score = 0.0
        if isinstance(fam_history, str):
            fam_score = 0.5 if fam_history.strip().lower() in ("yes", "y", "true") else 0.0
        elif fam_history:
            fam_score = 0.5
        else:
            for key in [
                "family_diabetes", "family_heart_disease",
                "family_stroke", "family_hypertension",
                "family_obesity", "family_kidney_disease",
                "family_thyroid", "family_depression",
            ]:
                if raw_dict.get(key):
                    fam_score += 0.5
        features["family_history_score"] = float(min(fam_score, 3.0))

        # 13. Blood Pressure
        bp_sys = raw_dict.get("blood_pressure_systolic")
        if bp_sys is None and raw_dict.get("blood_pressure"):
            try:
                bp_sys = float(raw_dict["blood_pressure"].split("/")[0])
            except Exception:
                bp_sys = 120.0
        features["blood_pressure_systolic"] = float(bp_sys if bp_sys else 120.0)

        # 14. Blood Sugar Fasting
        features["blood_sugar_fasting"] = float(
            FeaturePipeline._get_raw_value(
                raw_dict,
                "blood_sugar_fasting", "blood_sugar", default=95.0
            ) or 95.0
        )

        return features
