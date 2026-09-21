import argparse
import contextlib
import io
import tempfile
from pathlib import Path

import streamlit as st

from main import (
    BEST_MODEL_PKL,
    DECISION_THRESHOLD,
    FEATURE_ENGINEER_MODE,
    FEATURE_ENGINEER_PKL,
    INPUT_FIELDS,
    InputError,
    METRIC_NAMES,
    MODEL_MODE,
    PLOTS_DIR,
    PREPROCESSING_MODE,
    PREPROCESSOR_PKL,
    SELECTION_METRIC,
    _load_pickle,
    _read_records,
    cmd_train,
    predict_records,
)

st.set_page_config(page_title="Telco Churn Pipeline", page_icon="📊", layout="wide")


def _run_capturing(func, *args, **kwargs) -> str:
    """Run func and return whatever it printed to stdout."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func(*args, **kwargs)
    return buffer.getvalue()


# Train tab
def train_tab():
    st.subheader("Train models")

    with st.form("train_form"):
        preprocessing = st.selectbox(
            "Preprocessing",
            ["rerun", "load"],
            index=["rerun", "load"].index(PREPROCESSING_MODE),
            help="Rerun preprocessing or load the saved preprocessor pickle",
        )
        feature_engineering = st.selectbox(
            "Feature engineering",
            ["rerun", "load"],
            index=["rerun", "load"].index(FEATURE_ENGINEER_MODE),
            help="Rerun feature engineering or load the saved feature engineer pickle",
        )
        model = st.selectbox(
            "Model",
            ["rerun", "load"],
            index=["rerun", "load"].index(MODEL_MODE),
            help="Rerun model training or load the saved best model",
        )
        metric = st.selectbox(
            "Metric used to select the best model",
            list(METRIC_NAMES),
            index=list(METRIC_NAMES).index(SELECTION_METRIC),
        )
        show_plots = st.checkbox(
            "Show plots on screen",
            value=False,
            help="Opens a native window like --show-plots (plots are also shown below)",
        )
        submitted = st.form_submit_button("Train", type="primary")

    if not submitted:
        return

    args = argparse.Namespace(
        command=None,
        preprocessing=preprocessing,
        feature_engineering=feature_engineering,
        metric=metric,
        model=model,
        show_plots=show_plots,
    )

    with st.spinner("Training all models..."):
        try:
            output = _run_capturing(cmd_train, args)
        except (InputError, FileNotFoundError) as e:
            st.error(f"Error: {e}")
            return

    st.success("Training complete")
    with st.expander("Console output"):
        st.code(output, language="text")

    if show_plots:
        pngs = sorted(PLOTS_DIR.glob("*.png"))
        if pngs:
            st.subheader("Plots")
            for i in range(0, len(pngs), 2):
                cols = st.columns(2)
                for col, path in zip(cols, pngs[i : i + 2]):
                    col.image(str(path), caption=path.name)


# Predict tab
def _show_prediction(records, threshold):
    try:
        preprocessor = _load_pickle(PREPROCESSOR_PKL)
        feature_engineer = _load_pickle(FEATURE_ENGINEER_PKL)
        model = _load_pickle(BEST_MODEL_PKL)
        result = predict_records(records, preprocessor, feature_engineer, model, threshold)
    except (InputError, FileNotFoundError) as e:
        st.error(f"Error: {e}")
        return

    st.success(f"Model: {model.display_name} | Decision threshold: {threshold:.2f}")
    st.dataframe(result)

    for i, row in enumerate(result.itertuples(index=False), start=1):
        verdict = "likely to CHURN" if row.prediction == "Churn" else "likely to STAY"
        message = f"Customer #{i}: churn probability = {row.churn_probability:.1%} -> {verdict}"
        (st.error if row.prediction == "Churn" else st.info)(message)

    # Equivalent of --output
    st.download_button(
        "Download predictions as CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )


def _customer_form():
    with st.form("customer_form"):
        record = {}
        for field, spec in INPUT_FIELDS.items():
            kind = spec["type"]
            if kind == "choice":
                record[field] = st.selectbox(field, spec["choices"], key=field)
            elif kind == "flag":
                record[field] = st.selectbox(field, ["0", "1"], key=field, help=spec["hint"])
            elif kind == "int":
                record[field] = st.number_input(
                    field, min_value=0, step=1, value=0, key=field, help=spec["hint"]
                )
            elif spec["required"]:
                record[field] = st.number_input(
                    field, min_value=0.0, step=0.01, value=0.0, key=field, help=spec["hint"]
                )
            else:  # optional float (TotalCharges): blank = estimate
                record[field] = st.text_input(field, value="", key=field, help=spec["hint"])
        submitted = st.form_submit_button("Predict", type="primary")
    return record if submitted else None


def predict_tab():
    st.subheader("Predict churn")

    if not BEST_MODEL_PKL.exists():
        st.error(f"No trained model found ({BEST_MODEL_PKL.name}). Run the Train tab first.")
        return

    threshold = st.slider(
        "Decision threshold",
        0.0,
        1.0,
        float(DECISION_THRESHOLD),
        0.01,
        help="Probability at or above which a customer is predicted to churn",
    )
    source = st.radio("Input source", ["Form", "File upload", "JSON"], horizontal=True)

    if source == "Form":
        record = _customer_form()
        if record is not None:
            _show_prediction([record], threshold)
        return

    if source == "File upload":
        uploaded = st.file_uploader(
            "CSV or JSON file with one or more customers", type=["csv", "json"]
        )
        if uploaded is None:
            return
        if st.button("Predict file", type="primary"):
            suffix = Path(uploaded.name).suffix.lower() or ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            args = argparse.Namespace(
                json=None, file=tmp_path, interactive=False, threshold=threshold, output=None
            )
            try:
                records = _read_records(args)
            except InputError as e:
                st.error(f"Error: {e}")
                return
            _show_prediction(records, threshold)
        return

    # JSON source
    json_text = st.text_area(
        "JSON with one customer (object) or several (list of objects)",
        value='{"gender": "Female", "tenure": 3, ...}',
        height=160,
    )
    if st.button("Predict JSON", type="primary"):
        args = argparse.Namespace(
            json=json_text, file=None, interactive=False, threshold=threshold, output=None
        )
        try:
            records = _read_records(args)
        except InputError as e:
            st.error(f"Error: {e}")
            return
        _show_prediction(records, threshold)


st.title("Telco Churn Pipeline")

tab_train, tab_predict = st.tabs(["Train", "Predict"])
with tab_train:
    train_tab()
with tab_predict:
    predict_tab()