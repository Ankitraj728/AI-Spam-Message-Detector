"""Train and evaluate the SMS spam classifier.

Run ``python train_model.py`` with the full SMS Spam Collection, or pass
``--demo`` to train explicitly on the bundled sample data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SAMPLE_DATA = DATA_DIR / "sample_data.csv"
MODEL_FILE = PROJECT_DIR / "spam_model.joblib"
RESULTS_FILE = PROJECT_DIR / "results.txt"
EVALUATION_FILE = PROJECT_DIR / "evaluation_results.json"
CONFUSION_MATRIX_FILE = PROJECT_DIR / "confusion_matrix.png"
UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
CLASS_LABELS = ("ham", "spam")
MAX_MESSAGE_LENGTH = 5_000
ARTIFACT_FORMAT_VERSION = 7
RANDOM_STATE = 42
CV_FOLDS = 5
TARGET_SPAM_RECALL = 0.97

LABEL_VALUES = {
    "ham": "ham",
    "not spam": "ham",
    "not_spam": "ham",
    "legitimate": "ham",
    "legit": "ham",
    "normal": "ham",
    "0": "ham",
    "spam": "spam",
    "1": "spam",
}
LABEL_COLUMNS = ("label", "v1", "category", "class")
MESSAGE_COLUMNS = ("message", "v2", "text", "sms", "content")


class DatasetError(ValueError):
    """Raised when a dataset is missing or does not meet the training requirements."""


class ModelArtifactError(ValueError):
    """Raised when a saved model is missing, outdated, or invalid."""


def _resolve_dataset(
    dataset_path: str | Path | None, demo: bool, data_dir: Path
) -> tuple[Path, str]:
    if demo and dataset_path is not None:
        raise DatasetError("Choose either --demo or --dataset, not both.")

    if demo:
        path, dataset_type = data_dir / "sample_data.csv", "demo"
    elif dataset_path is not None:
        path = Path(dataset_path).expanduser()
        dataset_type = (
            "demo"
            if path.resolve() == (data_dir / "sample_data.csv").resolve()
            else "custom"
        )
    else:
        candidates = (
            data_dir / "spam.csv",
            data_dir / "SMSSpamCollection",
            data_dir / "SMSSpamCollection.txt",
            data_dir / "spam.tsv",
        )
        path = next(
            (candidate for candidate in candidates if candidate.is_file()), candidates[0]
        )
        dataset_type = "full"

    if not path.is_file():
        if dataset_type == "demo":
            raise DatasetError(f"Demo dataset not found: {path}")
        if dataset_path is not None:
            raise DatasetError(f"Dataset not found: {path}")
        raise DatasetError(
            "The full SMS Spam Collection dataset is missing. Place the Kaggle CSV at "
            f"{data_dir / 'spam.csv'} or the UCI tab-separated SMSSpamCollection file "
            f"at {data_dir / 'SMSSpamCollection'}, or pass --dataset PATH. "
            "For a clearly labeled demonstration only, pass --demo."
        )
    return path, dataset_type


def _read_frame(path: Path) -> pd.DataFrame:
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            first_line = file.readline()
    except UnicodeDecodeError:
        with path.open("r", encoding="latin-1") as file:
            first_line = file.readline()

    is_uci_tsv = path.suffix.lower() in {".tsv", ".txt"} or "\t" in first_line
    encodings = ("utf-8-sig", "latin-1")
    last_decode_error: UnicodeDecodeError | None = None

    for encoding in encodings:
        try:
            if is_uci_tsv:
                return pd.read_csv(
                    path,
                    sep="\t",
                    header=None,
                    names=["label", "message"],
                    encoding=encoding,
                    on_bad_lines="error",
                    quoting=csv.QUOTE_NONE,
                )
            return pd.read_csv(path, encoding=encoding, on_bad_lines="error")
        except UnicodeDecodeError as error:
            last_decode_error = error
        except pd.errors.ParserError as error:
            raise DatasetError(f"Could not parse dataset {path}: {error}") from error

    raise DatasetError(f"Could not decode dataset {path}: {last_decode_error}")


def _normalize_dataset(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    source_samples = len(frame)
    normalized_columns = {str(column).strip().casefold(): column for column in frame.columns}
    label_column = next(
        (normalized_columns[name] for name in LABEL_COLUMNS if name in normalized_columns),
        None,
    )
    message_column = next(
        (normalized_columns[name] for name in MESSAGE_COLUMNS if name in normalized_columns),
        None,
    )
    if label_column is None or message_column is None:
        raise DatasetError(
            f"Dataset {path} must have label and message columns. Supported pairs include "
            "'label,message', Kaggle's 'v1,v2', and 'category,message'."
        )

    data = frame[[label_column, message_column]].copy()
    missing_rows = data.isna().any(axis=1)
    if missing_rows.any():
        print(f"Removed {int(missing_rows.sum())} row(s) with missing labels or messages.")
        data = data.loc[~missing_rows].copy()

    data["label"] = data[label_column].astype(str).str.strip().str.casefold()
    data["message"] = data[message_column].astype(str).str.strip()
    blank_rows = data["label"].eq("") | data["message"].eq("")
    if blank_rows.any():
        print(f"Removed {int(blank_rows.sum())} row(s) with blank labels or messages.")
        data = data.loc[~blank_rows].copy()

    unexpected_labels = sorted(set(data["label"]) - LABEL_VALUES.keys())
    if unexpected_labels:
        shown = ", ".join(repr(label) for label in unexpected_labels[:10])
        raise DatasetError(
            f"Dataset {path} contains unexpected labels: {shown}. "
            "Expected ham/spam (or recognized equivalents such as 0/1)."
        )

    data["label"] = data["label"].map(LABEL_VALUES)
    source_class_counts = {
        label: int(count) for label, count in data["label"].value_counts().items()
    }
    message_keys = data["message"]
    conflicting_labels = (
        data.assign(_message_key=message_keys)
        .groupby("_message_key")["label"]
        .nunique()
        .gt(1)
    )
    if conflicting_labels.any():
        raise DatasetError(
            f"Dataset {path} has {int(conflicting_labels.sum())} message text(s) "
            "with conflicting ham/spam labels."
        )

    duplicate_rows = int(
        data.assign(_message_key=message_keys)
        .duplicated(subset=["label", "_message_key"])
        .sum()
    )
    if duplicate_rows:
        print(f"Removed {duplicate_rows} duplicate row(s).")
        data = data.assign(_message_key=message_keys).drop_duplicates(
            subset=["label", "_message_key"]
        )
        data = data.drop(columns="_message_key")

    data = data[["message", "label"]].reset_index(drop=True)
    if data.empty:
        raise DatasetError(f"Dataset {path} has no usable messages.")
    if set(data["label"]) != set(CLASS_LABELS):
        raise DatasetError("Dataset must contain both ham and spam examples.")
    data.attrs["source_samples"] = source_samples
    data.attrs["duplicates_removed"] = duplicate_rows
    data.attrs["source_class_counts"] = source_class_counts
    data.attrs["class_counts"] = {
        label: int(count) for label, count in data["label"].value_counts().items()
    }
    return data


def load_data(
    dataset_path: str | Path | None = None,
    *,
    demo: bool = False,
    data_dir: str | Path = DATA_DIR,
) -> pd.DataFrame:
    """Load and validate either the full corpus or explicitly requested demo data."""
    data_directory = Path(data_dir)
    path, _ = _resolve_dataset(dataset_path, demo, data_directory)
    return _normalize_dataset(_read_frame(path), path)


def validate_message(message: str) -> str:
    """Return a trimmed message or raise a helpful validation error."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Enter a message before analyzing it.")
    cleaned = message.strip()
    if len(cleaned) > MAX_MESSAGE_LENGTH:
        raise ValueError(
            f"Messages must be {MAX_MESSAGE_LENGTH:,} characters or fewer."
        )
    return cleaned


def predict_messages(
    model: Pipeline, messages: list[str]
) -> list[tuple[str, float | None]]:
    """Predict multiple messages, returning probability only when it is valid."""
    cleaned = [validate_message(message) for message in messages]
    if not cleaned:
        raise ValueError("Enter at least one message before analyzing.")
    classes = [str(label).casefold() for label in model.classes_]
    if "spam" not in classes:
        raise ValueError("The loaded model does not contain the expected ham/spam classes.")
    spam_probabilities: np.ndarray | None = None
    if callable(getattr(model, "predict_proba", None)):
        spam_probabilities = model.predict_proba(cleaned)[:, classes.index("spam")]
    threshold = getattr(model, "spam_decision_threshold_", None)
    if spam_probabilities is not None and threshold is not None:
        predictions = np.where(spam_probabilities >= threshold, "spam", "ham")
    else:
        predictions = model.predict(cleaned)
    if (
        any(str(label).casefold() not in CLASS_LABELS for label in predictions)
    ):
        raise ValueError("The loaded model does not contain the expected ham/spam classes.")

    probabilities: list[float | None] = [None] * len(cleaned)
    if (
        spam_probabilities is not None
        and getattr(model, "spam_probability_calibrated_", False)
    ):
        for index, probability in enumerate(spam_probabilities):
            value = float(probability)
            if math.isfinite(value) and 0.0 <= value <= 1.0:
                probabilities[index] = value
    return [
        (str(label).casefold(), probabilities[index])
        for index, label in enumerate(predictions)
    ]


def predict_message(model: Pipeline, message: str) -> tuple[str, float | None]:
    """Predict one message and return its label and valid spam probability, if available."""
    return predict_messages(model, [message])[0]


def _build_models(random_state: int) -> dict[str, Pipeline]:
    def pipeline(classifier: Any) -> Pipeline:
        return Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            (
                                "word_tfidf",
                                TfidfVectorizer(
                                    lowercase=True,
                                    stop_words=None,
                                    ngram_range=(1, 2),
                                ),
                            ),
                            (
                                "char_tfidf",
                                TfidfVectorizer(
                                    lowercase=True,
                                    analyzer="char_wb",
                                    ngram_range=(3, 5),
                                    min_df=2,
                                    max_features=200_000,
                                    sublinear_tf=True,
                                ),
                            ),
                        ]
                    ),
                ),
                ("classifier", classifier),
            ]
        )

    return {
        "Logistic Regression": pipeline(
            LogisticRegression(max_iter=1_000, random_state=random_state)
        ),
        "Multinomial Naive Bayes": pipeline(MultinomialNB()),
        "Linear SVM": pipeline(
            CalibratedClassifierCV(
                estimator=LinearSVC(random_state=random_state),
                method="sigmoid",
                cv=3,
            )
        ),
    }


def _model_parameter_grids() -> dict[str, dict[str, list[float]]]:
    return {
        "Logistic Regression": {"classifier__C": [0.1, 1.0, 10.0]},
        "Multinomial Naive Bayes": {"classifier__alpha": [0.1, 0.5, 1.0]},
        "Linear SVM": {
            "classifier__estimator__C": [0.1, 1.0, 10.0],
        },
    }


def _select_model_name(comparison: dict[str, Any]) -> str:
    """Use one-standard-error CV eligibility, then favor spam recall."""
    best_f1_name = max(
        comparison,
        key=lambda name: comparison[name]["cross_validation_f1"],
    )
    best_f1 = comparison[best_f1_name]["cross_validation_f1"]
    standard_error = comparison[best_f1_name].get("cross_validation_f1_standard_error", 0.0)
    minimum_f1 = best_f1 - standard_error
    eligible = [
        name
        for name, result in comparison.items()
        if result["cross_validation_f1"] >= minimum_f1
    ]
    return max(
        eligible,
        key=lambda name: (
            comparison[name].get("cross_validation_recall", 0.0),
            comparison[name]["cross_validation_f1"],
        ),
    )


def _score_model(
    model: Pipeline,
    x_test: pd.Series,
    y_test: pd.Series,
    *,
    decision_threshold: float | None = None,
) -> tuple[dict[str, float], list[list[int]], str]:
    if decision_threshold is None:
        decision_threshold = getattr(model, "spam_decision_threshold_", None)
    if decision_threshold is None:
        predictions = model.predict(x_test)
    else:
        classes = [str(label).casefold() for label in model.classes_]
        spam_scores = model.predict_proba(x_test)[:, classes.index("spam")]
        predictions = np.where(spam_scores >= decision_threshold, "spam", "ham")
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, pos_label="spam", zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, pos_label="spam", zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, pos_label="spam", zero_division=0)),
    }
    matrix = confusion_matrix(y_test, predictions, labels=list(CLASS_LABELS)).tolist()
    report = classification_report(
        y_test,
        predictions,
        labels=list(CLASS_LABELS),
        target_names=["Ham", "Spam"],
        zero_division=0,
    )
    return metrics, matrix, report


def _tune_spam_threshold(
    model: Pipeline,
    x_train: pd.Series,
    y_train: pd.Series,
    cross_validation: StratifiedKFold,
    parameter_grid: dict[str, list[float]],
    target_recall: float = TARGET_SPAM_RECALL,
) -> dict[str, float]:
    """Choose a threshold from nested training-only out-of-fold model scores."""
    spam_scores = np.empty(len(y_train), dtype=float)
    for training_indices, validation_indices in cross_validation.split(
        x_train, y_train
    ):
        fold_labels = y_train.iloc[training_indices]
        if min(fold_labels.value_counts().get(label, 0) for label in CLASS_LABELS) < (
            cross_validation.n_splits
        ):
            raise DatasetError(
                "Not enough examples in each training fold for nested "
                "cross-validation while tuning the Spam decision threshold."
            )
        threshold_search = GridSearchCV(
            estimator=model,
            param_grid=parameter_grid,
            scoring=make_scorer(f1_score, pos_label="spam", zero_division=0),
            cv=cross_validation,
            n_jobs=-1,
            error_score="raise",
        )
        try:
            threshold_search.fit(
                x_train.iloc[training_indices],
                fold_labels,
            )
        except ValueError as error:
            raise DatasetError(
                f"Could not tune the Spam decision threshold: {error}"
            ) from error
        fold_model = threshold_search.best_estimator_
        classes = [str(label).casefold() for label in fold_model.classes_]
        spam_scores[validation_indices] = fold_model.predict_proba(
            x_train.iloc[validation_indices]
        )[:, classes.index("spam")]
    precision, recall, thresholds = precision_recall_curve(
        y_train,
        spam_scores,
        pos_label="spam",
    )
    eligible = np.flatnonzero(recall[:-1] >= target_recall)
    if eligible.size == 0:
        raise DatasetError(
            f"Training cross-validation could not meet the target Spam recall "
            f"of {target_recall:.1%}."
        )
    index = int(eligible[-1])
    return {
        "threshold": float(thresholds[index]),
        "cross_validation_precision": float(precision[index]),
        "cross_validation_recall": float(recall[index]),
        "target_recall": target_recall,
    }


def _write_confusion_matrix(matrix: list[list[int]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=["Ham", "Spam"])
    axis.set_yticks([0, 1], labels=["Ham", "Spam"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Confusion Matrix")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, matrix[row][column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _format_results(metadata: dict[str, Any], comparison: dict[str, Any]) -> str:
    evaluation = metadata["evaluation"]
    lines = [
        "AI Spam Message Detector - Reproducible Evaluation",
        f"Dataset: {metadata['dataset_name']}",
        f"Dataset type: {metadata['dataset_type'].upper()}",
        f"Dataset file: {metadata['dataset_file']}",
        f"Dataset source: {metadata['dataset_source']}",
        f"Source dataset rows: {metadata['source_samples']}",
        f"Source class counts before deduplication: {metadata['source_class_counts']}",
        f"Duplicate messages removed: {metadata['duplicates_removed']}",
        f"Usable unique messages: {metadata['total_samples']}",
        f"Unique class counts: {metadata['class_counts']}",
        f"Unique ham messages: {metadata['class_counts']['ham']}",
        f"Unique spam messages: {metadata['class_counts']['spam']}",
        f"Python version: {metadata['software_versions']['python']}",
        f"scikit-learn version: {metadata['software_versions']['scikit_learn']}",
        f"Training samples: {metadata['training_samples']}",
        f"Testing samples: {metadata['testing_samples']}",
        f"Cross-validation folds (training only): {metadata['cv_folds']}",
        f"Random seed: {metadata['random_state']}",
        f"Selected model: {metadata['model_name']}",
        f"Spam decision threshold ({metadata['spam_decision']['threshold_basis']}): "
        f"{metadata['spam_decision']['threshold']:.4f}",
        f"Training CV Spam recall at threshold: "
        f"{metadata['spam_decision']['cross_validation_recall']:.4f}",
        f"Selection rule: {metadata['selection_rule']}",
        "Positive class: Spam",
        "",
        "Model selection (stratified cross-validation on training data only):",
    ]
    for name, result in comparison.items():
        lines.append(
            f"- {name}: mean CV Spam F1={result['cross_validation_f1']:.4f}; "
            f"mean CV Spam recall={result['cross_validation_recall']:.4f}; "
            f"best parameters={result['best_parameters']}"
        )
    lines.append(
        "Selection rule: highest CV Spam F1 defines the best score; models within "
        "one standard error are treated as statistically comparable, and the one "
        "with the highest training-only CV Spam recall is selected."
    )
    lines.extend(
        [
            "",
            "Selected model metrics (computed from the held-out test split):",
            "The selected model uses the recall-oriented threshold tuned with nested "
            "training-only out-of-fold model scores; alternative-model test metrics "
            "use each classifier's default prediction threshold.",
            f"Accuracy: {evaluation['accuracy']:.4f}",
            f"Precision (Spam): {evaluation['precision']:.4f}",
            f"Recall (Spam): {evaluation['recall']:.4f}",
            f"F1-score (Spam): {evaluation['f1']:.4f}",
            "",
            "Confusion matrix (rows = actual [Ham, Spam], columns = predicted [Ham, Spam]):",
            str(evaluation["confusion_matrix"]),
            "False positives (legitimate messages labeled Spam): "
            f"{evaluation['confusion_matrix'][0][1]}.",
            "False negatives (Spam messages labeled legitimate): "
            f"{evaluation['confusion_matrix'][1][0]}.",
            "",
            "Model comparison on the same held-out test split (not used for selection):",
            "CV metrics are out-of-fold results within the training partition. "
            "No in-sample training score is presented as a generalization estimate.",
        ]
    )
    for name, result in comparison.items():
        metrics = result["metrics"]
        lines.append(
            f"- {name}: CV Spam F1={result['cross_validation_f1']:.4f}, "
            f"CV Spam recall={result['cross_validation_recall']:.4f}, "
            f"spam cutoff={result.get('spam_decision_threshold', 'default')}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )
    for name, result in comparison.items():
        lines.extend(
            [
                "",
                f"Classification report on held-out test data - {name}:",
                result["classification_report"],
            ]
        )
    if metadata["dataset_type"] == "demo":
        lines.extend(
            [
                "",
                "IMPORTANT: These are demo-only metrics from the bundled sample dataset.",
                "They are not final SMS Spam Collection performance results.",
            ]
        )
    return "\n".join(lines) + "\n"


def train_and_evaluate(
    dataset_path: str | Path | None = None,
    *,
    demo: bool = False,
    data_dir: str | Path = DATA_DIR,
    output_dir: str | Path = PROJECT_DIR,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    cv_folds: int = CV_FOLDS,
) -> dict[str, Any]:
    """Select models using training-only CV and evaluate them on the untouched test split."""
    data_directory = Path(data_dir)
    path, dataset_type = _resolve_dataset(dataset_path, demo, data_directory)
    data = _normalize_dataset(_read_frame(path), path)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if (
        min(data["label"].value_counts().get(label, 0) for label in CLASS_LABELS)
        < cv_folds + 1
    ):
        raise DatasetError(
            f"Training requires at least {cv_folds + 1} usable examples of each class "
            "for the stratified test split and cross-validation."
        )

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            data["message"],
            data["label"],
            test_size=test_size,
            random_state=random_state,
            stratify=data["label"],
        )
    except ValueError as error:
        raise DatasetError(
            f"Could not create the requested stratified split: {error}"
        ) from error
    if min(y_train.value_counts().get(label, 0) for label in CLASS_LABELS) < cv_folds:
        raise DatasetError(
            f"The training split needs at least {cv_folds} examples of each class "
            "for stratified cross-validation."
        )
    if min(y_test.value_counts().get(label, 0) for label in CLASS_LABELS) < 1:
        raise DatasetError("The test split must contain examples of both classes.")

    comparison: dict[str, Any] = {}
    fitted_models: dict[str, Pipeline] = {}
    models = _build_models(random_state)
    parameter_grids = _model_parameter_grids()
    cross_validation = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    for name, model in models.items():
        try:
            search = GridSearchCV(
                estimator=model,
                param_grid=parameter_grids[name],
                scoring={
                    "f1": make_scorer(
                        f1_score, pos_label="spam", zero_division=0
                    ),
                    "recall": make_scorer(
                        recall_score, pos_label="spam", zero_division=0
                    ),
                },
                cv=cross_validation,
                n_jobs=-1,
                refit="f1",
                error_score="raise",
            )
            search.fit(x_train, y_train)
        except ValueError as error:
            raise DatasetError(f"Could not train {name} on this dataset: {error}") from error
        fitted_model = search.best_estimator_
        metrics, matrix, report = _score_model(fitted_model, x_test, y_test)
        fitted_models[name] = fitted_model
        best_index = search.best_index_
        comparison[name] = {
            "cross_validation_f1": float(search.cv_results_["mean_test_f1"][best_index]),
            "cross_validation_f1_standard_error": float(
                search.cv_results_["std_test_f1"][best_index] / (cv_folds**0.5)
            ),
            "cross_validation_recall": float(
                search.cv_results_["mean_test_recall"][best_index]
            ),
            "best_parameters": search.best_params_,
            "metrics": metrics,
            "confusion_matrix": matrix,
            "classification_report": report,
        }

    selected_name = _select_model_name(comparison)
    selected_model = fitted_models[selected_name]
    spam_decision = _tune_spam_threshold(
        _build_models(random_state)[selected_name].set_params(
            **comparison[selected_name]["best_parameters"]
        ),
        x_train,
        y_train,
        cross_validation,
        parameter_grids[selected_name],
    )
    selected_model.spam_decision_threshold_ = spam_decision["threshold"]
    probability_calibrated = selected_name == "Linear SVM"
    selected_model.spam_probability_calibrated_ = probability_calibrated
    spam_decision["threshold_basis"] = (
        "calibrated probability"
        if probability_calibrated
        else "uncalibrated model score"
    )
    selected_metrics, selected_matrix, selected_report = _score_model(
        selected_model,
        x_test,
        y_test,
        decision_threshold=spam_decision["threshold"],
    )
    comparison[selected_name]["metrics"] = selected_metrics
    comparison[selected_name]["confusion_matrix"] = selected_matrix
    comparison[selected_name]["classification_report"] = selected_report
    comparison[selected_name]["spam_decision_threshold"] = spam_decision["threshold"]
    selected_evaluation = comparison[selected_name]["metrics"].copy()
    selected_evaluation["confusion_matrix"] = comparison[selected_name][
        "confusion_matrix"
    ]
    selected_evaluation["classification_report"] = comparison[selected_name][
        "classification_report"
    ]
    dataset_name = {
        "demo": "Bundled sample dataset (demo only)",
        "full": "SMS Spam Collection",
        "custom": "User-specified dataset",
    }[dataset_type]
    dataset_source = (
        UCI_DATASET_URL
        if dataset_type == "full"
        else "Bundled demo dataset; not the UCI collection"
        if dataset_type == "demo"
        else "User-specified dataset; verify its source"
    )
    train_test_overlap = len(set(x_train).intersection(x_test))
    metadata: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "dataset_file": path.name,
        "dataset_source": dataset_source,
        "source_samples": int(data.attrs["source_samples"]),
        "duplicates_removed": int(data.attrs["duplicates_removed"]),
        "total_samples": len(data),
        "source_class_counts": data.attrs["source_class_counts"],
        "class_counts": data.attrs["class_counts"],
        "training_samples": len(x_train),
        "testing_samples": len(x_test),
        "training_class_counts": {
            label: int(count) for label, count in y_train.value_counts().items()
        },
        "testing_class_counts": {
            label: int(count) for label, count in y_test.value_counts().items()
        },
        "random_state": random_state,
        "test_size": test_size,
        "cv_folds": cv_folds,
        "software_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "positive_class": "spam",
        "class_labels": list(CLASS_LABELS),
        "text_preprocessing": {
            "vectorizer": "Word and character TF-IDF",
            "lowercase": True,
            "stop_words": None,
            "ngram_range": [1, 2],
            "character_ngram_range": [3, 5],
            "character_max_features": 200_000,
        },
        "model_name": selected_name,
        "spam_probability_calibrated": probability_calibrated,
        "probability_calibration": (
            "Sigmoid calibration with CalibratedClassifierCV"
            if probability_calibrated
            else None
        ),
        "spam_decision": spam_decision,
        "selection_rule": (
            "Highest mean Spam F1 defines the best score; candidates within one "
            "standard error on stratified training-only cross-validation are treated "
            "as statistically comparable, then highest Spam recall is selected. "
            "The held-out test split is never used for selection."
        ),
        "data_leakage_checks": {
            "duplicate_messages_removed_before_split": True,
            "train_test_message_overlap": train_test_overlap,
            "tfidf_is_inside_pipeline": all(
                isinstance(model.named_steps.get("features"), FeatureUnion)
                and all(
                    isinstance(vectorizer, TfidfVectorizer)
                    for _, vectorizer in model.named_steps["features"].transformer_list
                )
                for model in fitted_models.values()
            ),
            "cross_validation_uses_training_partition_only": True,
            "threshold_uses_nested_training_only_cross_validation": True,
            "test_data_used_for_model_selection": False,
        },
        "evaluation": selected_evaluation,
        "comparison": comparison,
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifact = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "model": selected_model,
        "metadata": metadata,
    }
    temporary_model_path = destination / ".spam_model.joblib.tmp"
    joblib.dump(artifact, temporary_model_path)
    temporary_model_path.replace(destination / MODEL_FILE.name)
    (destination / RESULTS_FILE.name).write_text(
        _format_results(metadata, comparison), encoding="utf-8"
    )
    (destination / EVALUATION_FILE.name).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_confusion_matrix(
        selected_evaluation["confusion_matrix"],
        destination / CONFUSION_MATRIX_FILE.name,
    )
    print(_format_results(metadata, comparison), end="")
    print(f"Saved model to {destination / MODEL_FILE.name}")
    print(f"Saved evaluation results to {destination / EVALUATION_FILE.name}")
    print(f"Saved confusion matrix to {destination / CONFUSION_MATRIX_FILE.name}")
    return metadata


def load_model_artifact(model_path: str | Path) -> tuple[Pipeline, dict[str, Any]]:
    """Load and validate a locally trained project artifact."""
    path = Path(model_path)
    if not path.is_file():
        raise ModelArtifactError(
            f"Model artifact not found: {path}. Train it with `python train_model.py`."
        )
    try:
        artifact = joblib.load(path)
    except (OSError, EOFError, ValueError, TypeError, ImportError, pickle.UnpicklingError) as error:
        raise ModelArtifactError(
            f"Could not read model artifact {path}. Retrain it with `python train_model.py`."
        ) from error

    if (
        not isinstance(artifact, dict)
        or artifact.get("format_version") != ARTIFACT_FORMAT_VERSION
    ):
        raise ModelArtifactError(
            "The saved model artifact is outdated or invalid. Retrain it with "
            "`python train_model.py` (or `python train_model.py --demo` for a demo)."
        )
    model = artifact.get("model")
    metadata = artifact.get("metadata")
    evaluation = metadata.get("evaluation") if isinstance(metadata, dict) else None
    valid_metadata = (
        isinstance(metadata, dict)
        and metadata.get("dataset_type") in {"full", "demo", "custom"}
        and metadata.get("positive_class") == "spam"
        and isinstance(metadata.get("model_name"), str)
        and isinstance(metadata.get("training_samples"), int)
        and isinstance(metadata.get("testing_samples"), int)
        and isinstance(metadata.get("total_samples"), int)
        and isinstance(metadata.get("source_samples"), int)
        and isinstance(metadata.get("dataset_source"), str)
        and isinstance(metadata.get("duplicates_removed"), int)
        and isinstance(metadata.get("class_counts"), dict)
        and isinstance(metadata.get("cv_folds"), int)
        and isinstance(evaluation, dict)
        and all(
            isinstance(evaluation.get(metric), (int, float))
            for metric in ("accuracy", "precision", "recall", "f1")
        )
        and isinstance(evaluation.get("confusion_matrix"), list)
        and isinstance(metadata.get("comparison"), dict)
        and isinstance(metadata.get("spam_decision"), dict)
        and isinstance(metadata.get("spam_probability_calibrated"), bool)
        and isinstance(
            metadata["spam_decision"].get("threshold"), (int, float)
        )
        and 0.0 <= metadata["spam_decision"]["threshold"] <= 1.0
        and getattr(model, "spam_decision_threshold_", None)
        == metadata["spam_decision"]["threshold"]
        and getattr(model, "spam_probability_calibrated_", None)
        == metadata["spam_probability_calibrated"]
    )
    if (
        not isinstance(model, Pipeline)
        or not valid_metadata
        or not {"ham", "spam"}.issubset(
            {str(label).casefold() for label in model.classes_}
        )
    ):
        raise ModelArtifactError(
            "The saved model artifact is invalid. Retrain it with `python train_model.py`."
        )
    return model, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SMS spam message classifier.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", type=Path, help="Path to a supported full/custom dataset.")
    source.add_argument(
        "--demo",
        action="store_true",
        help="Explicitly train on the bundled sample data (demo results only).",
    )
    args = parser.parse_args()
    try:
        train_and_evaluate(args.dataset, demo=args.demo)
    except DatasetError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
