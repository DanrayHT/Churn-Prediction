import logging
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

logger = logging.getLogger(__name__)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")


# Figure helpers
def _create_figure(nrows, ncols, figsize, show):
    if show:
        import matplotlib.pyplot as plt

        return plt.subplots(nrows, ncols, figsize=figsize)
    fig = Figure(figsize=figsize)
    return fig, fig.subplots(nrows, ncols)


def _finish_figure(fig, path, show):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        import matplotlib.pyplot as plt

        plt.show()
        plt.close(fig)

# Base class
class Model(ABC):
    name = "base"
    display_name = "Base model"

    def __init__(self, random_state=42, threshold=0.5):
        self.random_state = random_state
        self.threshold = threshold  # probability threshold for predicting "Churn"
        self.estimator = self._build_estimator()
        self.feature_names = None
        self.metrics = {}
        self.fit_time = None
        self._y_true = None
        self._y_proba = None
        self._y_pred = None

    @abstractmethod
    def _build_estimator(self):
        """Return an unfitted scikit-learn estimator."""

    def _fit_estimator(self, X, y):
        self.estimator.fit(X, y)

    def fit(self, X_train, y_train):
        self.feature_names = list(X_train.columns)
        start = time.perf_counter()
        self._fit_estimator(X_train, y_train)
        self.fit_time = time.perf_counter() - start
        logger.info(f"Trained {self.display_name} in {self.fit_time:.2f}s")
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X, threshold=None):
        thr = self.threshold if threshold is None else threshold
        return (self.predict_proba(X)[:, 1] >= thr).astype(int)

    def evaluate(self, X_test, y_test):
        proba = self.predict_proba(X_test)[:, 1]
        pred = (proba >= self.threshold).astype(int)
        y_true = np.asarray(y_test)

        self._y_true, self._y_proba, self._y_pred = y_true, proba, pred
        self.metrics = {
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, proba),
        }
        logger.info(f"Evaluated {self.display_name} | {self.results_text()}")
        return self.metrics

    def results_text(self):
        return " | ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())

    def get_feature_importance(self):
        if hasattr(self.estimator, "feature_importances_"):
            values = self.estimator.feature_importances_
        elif hasattr(self.estimator, "coef_"):  # an RBF SVM falls through to the else branch
            values = np.ravel(self.estimator.coef_)
        else:
            return None
        return pd.Series(values, index=self.feature_names)

    # Plots
    def _check_evaluated(self):
        if self._y_true is None:
            raise RuntimeError(f"{self.display_name}: call evaluate() before plotting")

    def plot_confusion_matrix(self, ax):
        self._check_evaluated()
        ConfusionMatrixDisplay.from_predictions(
            self._y_true,
            self._y_pred,
            display_labels=["Stay", "Churn"],
            ax=ax,
            colorbar=False,
            cmap="Blues",
        )
        ax.set_title(f"Confusion matrix (threshold = {self.threshold})")

    def plot_roc_curve(self, ax, label=None):
        self._check_evaluated()
        fpr, tpr, _ = roc_curve(self._y_true, self._y_proba)
        auc_value = self.metrics["roc_auc"]
        ax.plot(fpr, tpr, lw=2, label=label or f"AUC = {auc_value:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="_nolegend_")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

    def plot_pr_curve(self, ax):
        self._check_evaluated()
        precision, recall, _ = precision_recall_curve(self._y_true, self._y_proba)
        ap = average_precision_score(self._y_true, self._y_proba)
        ax.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
        ax.axhline(self._y_true.mean(), color="k", ls="--", lw=1, label="Baseline")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall curve")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    def plot_feature_importance(self, ax, top_n=15):
        importance = self.get_feature_importance()
        if importance is None:
            ax.text(
                0.5,
                0.5,
                "Feature importance is not\navailable for this model",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Feature importance")
            ax.set_axis_off()
            return

        order = importance.abs().sort_values(ascending=False).index
        top = importance.reindex(order).head(top_n).iloc[::-1]
        signed = hasattr(self.estimator, "coef_")
        colors = ["tab:red" if v < 0 else "tab:blue" for v in top.values] if signed else "tab:blue"

        ax.barh(top.index, top.values, color=colors)
        ax.set_title(
            f"Top {len(top)} coefficients (blue = raises churn risk, red = lowers it)"
            if signed
            else f"Top {len(top)} feature importances"
        )
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.3)

    def plot_all(self, save_dir, show=False):
        self._check_evaluated()
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = _create_figure(2, 2, figsize=(14, 11), show=show)
        self.plot_confusion_matrix(axes[0, 0])
        self.plot_roc_curve(axes[0, 1])
        self.plot_pr_curve(axes[1, 0])
        self.plot_feature_importance(axes[1, 1])
        fig.suptitle(f"{self.display_name}\n{self.results_text()}", fontsize=12)

        path = save_dir / f"{self.name}_report.png"
        _finish_figure(fig, path, show)
        return path

    # Utilities
    def run(self, X_train, X_test, y_train, y_test, plot_dir=None, show=False):
        self.fit(X_train, y_train)
        self.evaluate(X_test, y_test)
        if plot_dir is not None:
            self.plot_all(plot_dir, show=show)
        return self.metrics

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self):
        metrics = self.metrics or "not evaluated"
        return f"<{self.__class__.__name__} name={self.name!r} metrics={metrics}>"


# Models
class LogisticRegressionModel(Model):
    name = "logistic_regression"
    display_name = "Logistic Regression"

    def _build_estimator(self):
        return LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=self.random_state
        )


class RandomForestModel(Model):
    name = "random_forest"
    display_name = "Random Forest"

    def _build_estimator(self):
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.random_state,
        )


class GradientBoostingModel(Model):
    name = "gradient_boosting"
    display_name = "Gradient Boosting"

    def _build_estimator(self):
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=self.random_state,
        )

    def _fit_estimator(self, X, y):
        # GradientBoosting has no class_weight, so balance the classes with sample_weight
        weights = compute_sample_weight("balanced", y)
        self.estimator.fit(X, y, sample_weight=weights)


class SVMModel(Model):
    name = "svm"
    display_name = "SVM (RBF)"

    def _build_estimator(self):
        return SVC(
            kernel="rbf",
            C=1.0,
            probability=True,  # needed for predict_proba and ROC-AUC
            class_weight="balanced",
            random_state=self.random_state,
        )


MODEL_REGISTRY = {
    cls.name: cls
    for cls in (LogisticRegressionModel, RandomForestModel, GradientBoostingModel, SVMModel)
}


# Evaluator
class Evaluator:
    def __init__(self, models, metric="roc_auc"):
        metric = str(metric).strip().lower()
        if metric not in METRIC_NAMES:
            raise ValueError(f"Invalid metric '{metric}'. Choose one of: {', '.join(METRIC_NAMES)}")
        if not models:
            raise ValueError("No models to evaluate")
        not_ready = [m.display_name for m in models if not m.metrics]
        if not_ready:
            raise ValueError(f"Models not evaluated yet: {', '.join(not_ready)}")

        self.models = list(models)
        self.metric = metric
        self.best_model = None

    def summary(self):
        rows = {m.display_name: {**m.metrics, "fit_time_s": m.fit_time} for m in self.models}
        table = pd.DataFrame.from_dict(rows, orient="index")
        sort_keys = [self.metric] + [k for k in ("roc_auc", "f1") if k != self.metric]
        return table.sort_values(sort_keys, ascending=False)

    def select_best(self):
        best_name = self.summary().index[0]
        self.best_model = next(m for m in self.models if m.display_name == best_name)
        score = self.best_model.metrics[self.metric]
        logger.info(f"Best model by {self.metric}: {best_name} ({score:.4f})")
        return self.best_model

    def plot_comparison(self, save_dir, show=False):
        best = self.best_model or self.select_best()
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        table = self.summary()

        fig, (ax_bar, ax_roc) = _create_figure(1, 2, figsize=(16, 6), show=show)
        
        metrics = list(METRIC_NAMES)
        positions = np.arange(len(table))
        width = 0.8 / len(metrics)
        for i, metric in enumerate(metrics):
            offset = (i - (len(metrics) - 1) / 2) * width
            ax_bar.bar(positions + offset, table[metric], width, label=metric)
        ax_bar.set_xticks(positions)
        ax_bar.set_xticklabels(table.index, rotation=15)
        ax_bar.set_ylim(0, 1.15)
        ax_bar.set_ylabel("Score")
        ax_bar.set_title(f"Metrics (test set) - best by {self.metric}: {best.display_name}")
        ax_bar.legend(loc="upper center", ncol=len(metrics), fontsize=8)
        ax_bar.grid(axis="y", alpha=0.3)

        for m in self.models:
            m.plot_roc_curve(ax_roc, label=f"{m.display_name} (AUC={m.metrics['roc_auc']:.3f})")
        ax_roc.set_title("ROC curves (test set)")

        path = save_dir / "model_comparison.png"
        _finish_figure(fig, path, show)
        return path