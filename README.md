# Telco Customer Churn Prediction

A machine-learning pipeline that predicts whether a telecom customer is likely to **churn**. It trains several models on the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset, picks the best one by a configurable metric, and lets you make predictions — either from the command line or through a Streamlit web app.

> **Live demo:** try the hosted app at
> <https://churn-prediction-ickvyn82vhxmaam9rby3fu.streamlit.app/>

---

## Features

- **End-to-end pipeline** — load → clean/preprocess → feature engineering → train → evaluate → predict.
- **Four models compared automatically:** Logistic Regression, Random Forest, Gradient Boosting, and SVM (RBF). All are class-balanced to handle the imbalanced churn target.
- **Best-model selection** by any of: `accuracy`, `precision`, `recall`, `f1`, or `roc_auc`.
- **Two ways to use it:** a CLI (`main.py`) and a Streamlit web app (`app.py`).
- **Reproducible & configurable** — every knob lives in `src/config.ini`; trained artifacts are pickled so you can skip retraining.
- **Auto-downloads the dataset** from Kaggle on first run if it isn't present locally.

---

## Project structure

```
.
├── app.py                 # Streamlit web app (Train + Predict tabs)
├── main.py                # CLI entry point (train / predict commands)
├── requirements.txt       # Python dependencies
├── data/
│   └── telco_churn.csv    # Raw dataset (auto-downloaded if missing)
├── logs/
│   └── app.log            # Runtime log file
├── save/                  # Saved artifacts
│   ├── preprocessor.pkl   # Fitted Preprocessor
│   ├── feature_engineer.pkl  # Fitted FeatureEngineer
│   ├── best_model.pkl     # Best trained model
│   └── plots/             # Generated PNG reports
└── src/
    ├── config.ini         # All configuration (paths, pipeline, training)
    ├── data_loader.py     # Loads / downloads the dataset
    ├── preprocessing.py   # Preprocessor + FeatureEngineer
    └── train.py           # Models, Evaluator, and plotting
```

---

## Installation

Requires **Python 3.9+**.

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `streamlit`, and `kagglehub` (used only to auto-download the dataset).

> **Note:** Auto-downloading from Kaggle requires a Kaggle account with API credentials configured (`kagglehub`). If you already have `data/telco_churn.csv`, no Kaggle setup is needed.

---

## Configuration

All settings live in [`src/config.ini`](src/config.ini):

```ini
[paths]
raw_data_path = data/telco_churn.csv
log_path      = logs/app.log
save_dir      = save

[pipeline]
target_col            = Churn
preprocessing_mode    = rerun     # default mode for preprocessing (rerun | load)
feature_engineer_mode = rerun     # default mode for feature engineering (rerun | load)
model_mode            = rerun     # default mode for model training (rerun | load)
test_size             = 0.2       # fraction held out as the test set
random_state          = 42        # seed for reproducibility

[training]
selection_metric   = roc_auc      # accuracy | precision | recall | f1 | roc_auc
decision_threshold = 0.5          # churn probability >= this => "Churn"
```

- **`rerun` vs `load`:** each pipeline stage can either re-run from scratch or load a previously saved pickle (`preprocessor.pkl`, `feature_engineer.pkl`, `best_model.pkl`). The defaults come from the ini file and can be overridden per run (see CLI flags below).
- **`selection_metric`:** which metric is used to choose the best model.
- **`decision_threshold`:** probability cutoff for labeling a customer as "Churn".

---

## How it works

1. **Load data** — reads `data/telco_churn.csv` (downloads from Kaggle if absent) and verifies the target column exists.
2. **Preprocessing** (`src/preprocessing.py`)
   - Converts `TotalCharges` to numeric, drops `customerID`, removes duplicates, and drops rows with missing values.
   - Encodes binary/service columns (`Yes/No`, `Male/Female`, etc.) as 0/1.
   - One-hot encodes the remaining categorical columns (e.g., `InternetService`, `Contract`, `PaymentMethod`).
   - Maps the target `Churn` from `Yes/No` to `1/0`.
3. **Feature engineering** — performs a stratified train/test split (`test_size=0.2`) and applies `StandardScaler` to numeric columns only (categoricals are left unscaled).
4. **Training & evaluation** (`src/train.py`) — trains all four models, computes accuracy / precision / recall / F1 / ROC-AUC on the test set, and saves per-model report plots (confusion matrix, ROC curve, precision-recall curve, feature importance) to `save/plots/`.
5. **Best model selection** — picks the top model by the configured metric and saves it to `save/best_model.pkl`.

### Models

| Model | Key settings |
| --- | --- |
| Logistic Regression | `max_iter=1000`, `class_weight="balanced"` |
| Random Forest | `n_estimators=300`, `min_samples_leaf=3`, `class_weight="balanced"`, `n_jobs=-1` |
| Gradient Boosting | `n_estimators=200`, `learning_rate=0.5→0.05`, `max_depth=3`, balanced sample weights |
| SVM (RBF) | `C=1.0`, `probability=True`, `class_weight="balanced"` |

---

## Using the CLI (`main.py`)

### Train

```bash
# Default: rerun everything, select by roc_auc
python main.py

# Select a different metric and show plots on screen
python main.py --metric f1 --show-plots

# Reuse saved preprocessing / feature engineering (only retrain models)
python main.py --preprocessing load --feature-engineering load
```

Training options:

| Flag | Values | Description |
| --- | --- | --- |
| `--preprocessing` | `rerun`, `load` | Rerun or load the saved preprocessor |
| `--feature-engineering` | `rerun`, `load` | Rerun or load the saved feature engineer |
| `--model` | `rerun`, `load` | Retrain models or just load the saved best model |
| `--metric` | `accuracy`, `precision`, `recall`, `f1`, `roc_auc` | Metric used to select the best model |
| `--show-plots` | — | Display plots in a window (they're always also saved as PNG) |

### Predict

```bash
# Interactive: type each field by hand
python main.py predict

# From a CSV or JSON file with one or more customers
python main.py predict --file customers.csv

# From an inline JSON string
python main.py predict --json '{"gender": "Female", "tenure": 3, ...}'

# Save results to a CSV and use a custom threshold
python main.py predict --file customers.csv --output predictions.csv --threshold 0.4
```

Prediction options:

| Flag | Description |
| --- | --- |
| `--json` | JSON string with one customer (object) or several (list of objects) |
| `--file` | Path to a `.csv` or `.json` file with one or more customers |
| `--interactive` | Enter each field by hand (default when neither `--json` nor `--file` is given) |
| `--threshold` | Probability at/above which a customer is predicted to churn (default from config) |
| `--output` | Save predictions to a CSV file (use with `--json` or `--file`) |

**Input fields:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, and optional `TotalCharges` (if omitted, it's estimated as `tenure × MonthlyCharges`).

---

## Using the web app (`app.py`)

```bash
streamlit run app.py
```

Then open <http://localhost:8501> in your browser. The app has two tabs:

- **Train** — choose the preprocessing / feature-engineering / model mode, the selection metric, and whether to show plots on screen. Clicking **Train** runs the full pipeline, shows the captured console output, and displays the generated report plots.
- **Predict** — set a decision-threshold slider and pick an input source (**Form**, **File upload**, or **JSON**). It returns each customer's churn probability and verdict (Churn / Stay), plus a button to download the results as CSV.

> The hosted version of this app is available at
> <https://churn-prediction-ickvyn82vhxmaam9rby3fu.streamlit.app/>.

---

## Outputs

After training, you'll find in `save/`:

- `preprocessor.pkl`, `feature_engineer.pkl`, `best_model.pkl` — reusable fitted artifacts.
- `plots/*.png` — per-model reports and a model-comparison chart.

Runtime details are logged to `logs/app.log`.

---

## Notes

- This project is intended for **testing and practice** on the specific Telco churn dataset.
- The pipeline is deterministic given the same data, seed (`random_state=42`), and configuration.