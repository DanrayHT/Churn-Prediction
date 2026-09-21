import argparse
import configparser
import json
import logging
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocessing import FeatureEngineer, Preprocessor
from src.train import METRIC_NAMES, MODEL_REGISTRY, Evaluator

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "src" / "config.ini"


def _load_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    # read() expects file paths, not file objects
    read_files = config.read(CONFIG_PATH, encoding="utf-8-sig")
    if not read_files:
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    return config


_config = _load_config()

LOG_PATH = (BASE_DIR / _config["paths"]["log_path"]).resolve()
SAVE_DIR = (BASE_DIR / _config["paths"]["save_dir"]).resolve()
TARGET_COL = _config["pipeline"]["target_col"]
TEST_SIZE = float(_config["pipeline"]["test_size"])
RANDOM_STATE = int(_config["pipeline"]["random_state"])

# Defaults from the ini file: "rerun"
PREPROCESSING_MODE = _config["pipeline"]["preprocessing_mode"].strip().lower()
FEATURE_ENGINEER_MODE = _config["pipeline"]["feature_engineer_mode"].strip().lower()
MODEL_MODE = _config.get("pipeline", "model_mode", fallback="rerun").strip().lower()

SELECTION_METRIC = _config.get("training", "selection_metric", fallback="roc_auc").strip().lower()
DECISION_THRESHOLD = _config.getfloat("training", "decision_threshold", fallback=0.5)

PREPROCESSOR_PKL = SAVE_DIR / "preprocessor.pkl"
FEATURE_ENGINEER_PKL = SAVE_DIR / "feature_engineer.pkl"
BEST_MODEL_PKL = SAVE_DIR / "best_model.pkl"
PLOTS_DIR = SAVE_DIR / "plots"

SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class InputError(ValueError):
    """Invalid user input. Reported as a short message instead of a traceback."""


# Prediction input fields
_YES_NO = ["Yes", "No"]
_YES_NO_NOINT = ["Yes", "No", "No internet service"]


def _choice(choices):
    return {"type": "choice", "choices": choices, "required": True}


INPUT_FIELDS = {
    "gender": _choice(["Male", "Female"]),
    "SeniorCitizen": {"type": "flag", "required": True, "hint": "0 = no, 1 = yes"},
    "Partner": _choice(_YES_NO),
    "Dependents": _choice(_YES_NO),
    "tenure": {
        "type": "int",
        "required": True,
        "hint": "months with the service, whole number >= 0",
    },
    "PhoneService": _choice(_YES_NO),
    "MultipleLines": _choice(["Yes", "No", "No phone service"]),
    "InternetService": _choice(["DSL", "Fiber optic", "No"]),
    "OnlineSecurity": _choice(_YES_NO_NOINT),
    "OnlineBackup": _choice(_YES_NO_NOINT),
    "DeviceProtection": _choice(_YES_NO_NOINT),
    "TechSupport": _choice(_YES_NO_NOINT),
    "StreamingTV": _choice(_YES_NO_NOINT),
    "StreamingMovies": _choice(_YES_NO_NOINT),
    "Contract": _choice(["Month-to-month", "One year", "Two year"]),
    "PaperlessBilling": _choice(_YES_NO),
    "PaymentMethod": _choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    ),
    "MonthlyCharges": {"type": "float", "required": True, "hint": "monthly bill, number >= 0"},
    "TotalCharges": {
        "type": "float",
        "required": False,
        "hint": "total billed so far, leave blank to estimate as tenure x MonthlyCharges",
    },
}
_CANONICAL = {name.lower(): name for name in INPUT_FIELDS}


# Input validation
def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return bool(pd.isna(value))


def _hint(field: str) -> str:
    spec = INPUT_FIELDS[field]
    if spec["type"] == "choice":
        return "choose one of: " + ", ".join(spec["choices"])
    return spec["hint"]


def _coerce(field: str, value):
    spec = INPUT_FIELDS[field]
    kind = spec["type"]
    text = str(value).strip()
    error = InputError(f"Invalid value for '{field}': {value!r} ({_hint(field)})")

    if kind == "choice":
        lookup = {c.lower(): c for c in spec["choices"]}  # case-insensitive match
        if text.lower() not in lookup:
            raise error
        return lookup[text.lower()]

    if kind == "flag":
        key = text.lower()
        if key in ("0.0", "1.0"):
            key = key[0]
        mapping = {"0": 0, "1": 1, "no": 0, "yes": 1, "false": 0, "true": 1}
        if key not in mapping:
            raise error
        return mapping[key]

    try:
        number = float(text)
    except ValueError:
        raise error from None
    if not math.isfinite(number) or number < 0:
        raise error
    if kind == "int":
        if not number.is_integer():
            raise error
        return int(number)
    return number


def _validate_record(record: dict, index: int) -> dict:
    record = {_CANONICAL.get(str(k).strip().lower()): v for k, v in record.items()}
    record.pop(None, None)

    missing = [f for f, s in INPUT_FIELDS.items() if s["required"] and _is_blank(record.get(f))]
    if missing:
        raise InputError(f"Customer #{index}: missing required fields: {', '.join(missing)}")

    clean = {}
    for field in INPUT_FIELDS:
        if _is_blank(record.get(field)):
            continue
        try:
            clean[field] = _coerce(field, record[field])
        except InputError as e:
            raise InputError(f"Customer #{index}: {e}") from None

    if "TotalCharges" not in clean:
        clean["TotalCharges"] = round(clean["tenure"] * clean["MonthlyCharges"], 2)
    return clean


def build_input_frame(records: list) -> pd.DataFrame:
    rows = [_validate_record(r, i) for i, r in enumerate(records, start=1)]
    return pd.DataFrame(rows, columns=list(INPUT_FIELDS))


# Input sources: --json, --file, interactive
def _read_records(args) -> list:
    try:
        if args.json:
            data = json.loads(args.json)
        else:
            path = Path(args.file)
            if not path.exists():
                raise InputError(f"File not found: {path}")
            if path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            else:
                data = pd.read_csv(path).to_dict("records")
    except json.JSONDecodeError as e:
        raise InputError(f"Invalid JSON: {e}") from None

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data or not all(isinstance(r, dict) for r in data):
        raise InputError("Input must be a JSON object or a list of JSON objects")
    return data


def _prompt_record() -> dict:
    print("\nEnter customer details (for choices, type the option number or the value):")
    record = {}
    for field, spec in INPUT_FIELDS.items():
        if spec["type"] == "choice":
            options = "  ".join(f"[{i}] {c}" for i, c in enumerate(spec["choices"], start=1))
            label = f"{field}: {options}"
        else:
            label = f"{field} ({spec['hint']})"

        while True:
            raw = input(f"{label}\n  > ").strip()
            if raw == "" and not spec["required"]:
                break
            if spec["type"] == "choice" and raw.isdigit() and 1 <= int(raw) <= len(spec["choices"]):
                raw = spec["choices"][int(raw) - 1]
            try:
                record[field] = _coerce(field, raw)
                break
            except InputError as e:
                print(f"  ! {e}")
    return record


# Data pipeline
def _load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}. Run training first: python main.py")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_datasets(args):
    df = load_data()

    if args.preprocessing == "load":
        preprocessor = _load_pickle(PREPROCESSOR_PKL)
        df_clean = preprocessor.clean(df)
        X = preprocessor.transform(df_clean)
        y = df_clean[TARGET_COL].map({"Yes": 1, "No": 0})
    else:
        preprocessor = Preprocessor(target=TARGET_COL)
        X, y = preprocessor.fit_transform(df)
        with open(PREPROCESSOR_PKL, "wb") as f:
            pickle.dump(preprocessor, f)

    # Feature engineering
    if args.feature_engineering == "load":
        feature_engineer = _load_pickle(FEATURE_ENGINEER_PKL)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=feature_engineer.test_size,
            random_state=feature_engineer.random_state,
        )
        X_train = feature_engineer.transform(X_train)
        X_test = feature_engineer.transform(X_test)
    else:
        feature_engineer = FeatureEngineer(test_size=TEST_SIZE, random_state=RANDOM_STATE)
        X_train, X_test, y_train, y_test = feature_engineer.fit_transform(
            X, y, categorical_columns=preprocessor.categorical_feature_columns
        )
        with open(FEATURE_ENGINEER_PKL, "wb") as f:
            pickle.dump(feature_engineer, f)

    return X_train, X_test, y_train, y_test


# Command: train
def cmd_train(args):
    if args.model == "load":
        best = _load_pickle(BEST_MODEL_PKL)
        print(f"Loaded best model: {best.display_name}") 
        print(f"Model loaded from: {BEST_MODEL_PKL}")
        if PLOTS_DIR.exists(): 
            pngs = sorted(PLOTS_DIR.glob("*.png"))
            if pngs: print(f"Plots available in: {PLOTS_DIR}") 
        return

    X_train, X_test, y_train, y_test = build_datasets(args)

    models = []
    for model_cls in MODEL_REGISTRY.values():
        model = model_cls(random_state=RANDOM_STATE, threshold=DECISION_THRESHOLD)
        print(f"Training: {model.display_name}")
        model.run(X_train, X_test, y_train, y_test, plot_dir=PLOTS_DIR, show=args.show_plots)
        models.append(model)

    evaluator = Evaluator(models, metric=args.metric)
    best = evaluator.select_best()
    evaluator.plot_comparison(PLOTS_DIR, show=args.show_plots)
    best.save(BEST_MODEL_PKL)

    print(f"\nResults on the test set (sorted by {args.metric}):")
    print(evaluator.summary().round(4).to_string())
    print(f"\nBest model by {args.metric}: {best.display_name}")
    print(f"  Model saved to: {BEST_MODEL_PKL}")
    print(f"  Plots saved to: {PLOTS_DIR}")


# Command: predict
def predict_records(records, preprocessor, feature_engineer, model, threshold) -> pd.DataFrame:
    raw = build_input_frame(records)
    X = feature_engineer.transform(preprocessor.transform(raw))
    proba = model.predict_proba(X)[:, 1]

    result = raw.copy()
    result["churn_probability"] = proba
    result["prediction"] = np.where(proba >= threshold, "Churn", "Stay")
    return result


def _print_predictions(result: pd.DataFrame, threshold: float):
    print(f"\nDecision threshold: {threshold:.2f}")
    for i, row in enumerate(result.itertuples(index=False), start=1):
        verdict = "likely to CHURN" if row.prediction == "Churn" else "likely to STAY"
        print(f"Customer #{i}: churn probability = {row.churn_probability:.1%} -> {verdict}")


def cmd_predict(args):
    preprocessor = _load_pickle(PREPROCESSOR_PKL)
    feature_engineer = _load_pickle(FEATURE_ENGINEER_PKL)
    model = _load_pickle(BEST_MODEL_PKL)
    print(f"Model: {model.display_name}")

    def run(records):
        return predict_records(records, preprocessor, feature_engineer, model, args.threshold)

    if args.json or args.file:
        result = run(_read_records(args))
        _print_predictions(result, args.threshold)
        if args.output:
            result.to_csv(args.output, index=False)
            print(f"\nPredictions saved to: {args.output}")
        return

    while True:
        result = run([_prompt_record()])
        _print_predictions(result, args.threshold)
        if input("\nPredict another customer? (y/n): ").strip().lower() not in ("y", "yes"):
            break


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telco churn pipeline")

    # Options for training
    parser.add_argument(
        "--preprocessing",
        choices=["rerun", "load"],
        default=PREPROCESSING_MODE,
        help="Rerun preprocessing or load the saved preprocessor pickle (default: %(default)s)",
    )
    parser.add_argument(
        "--feature-engineering",
        choices=["rerun", "load"],
        default=FEATURE_ENGINEER_MODE,
        help="Rerun feature engineering or load the saved feature engineer pickle "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        choices=["rerun", "load"],
        default=MODEL_MODE,
        help="Rerun model training or load the saved best model "
        "(default from config.ini: %(default)s)",
    )
    parser.add_argument(
        "--metric",
        choices=METRIC_NAMES,
        default=SELECTION_METRIC,
        help="Metric used to select the best model (default from config.ini: %(default)s)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots on screen (by default they are only saved as PNG)",
    )

    sub = parser.add_subparsers(dest="command")
    predict = sub.add_parser("predict", help="Predict churn for new customers with the best model")
    source = predict.add_mutually_exclusive_group()
    source.add_argument(
        "--json", help="JSON string with one customer (object) or several (list of objects)"
    )
    source.add_argument("--file", help="Path to a .csv or .json file with one or more customers")
    source.add_argument(
        "--interactive",
        action="store_true",
        help="Enter each field by hand (default when neither --json nor --file is given)",
    )
    predict.add_argument(
        "--threshold",
        type=float,
        default=DECISION_THRESHOLD,
        help="Probability at or above which a customer is predicted to churn "
        "(default from config.ini: %(default)s)",
    )
    predict.add_argument(
        "--output", help="Save predictions to a CSV file (use with --json or --file)"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.command == "predict":
            cmd_predict(args)
        else:
            cmd_train(args)
    except (InputError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        print("\nExited")
        sys.exit(0)


if __name__ == "__main__":
    main()