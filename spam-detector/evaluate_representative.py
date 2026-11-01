"""Evaluate the saved classifier on a separate, labeled diagnostic message set."""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from train_model import (
    CLASS_LABELS,
    DATA_DIR,
    MODEL_FILE,
    PROJECT_DIR,
    _build_models,
    _score_model,
    load_data,
    load_model_artifact,
    predict_messages,
)

FIXTURE_FILE = PROJECT_DIR / "tests" / "fixtures" / "representative_messages.csv"
REPORT_FILE = PROJECT_DIR / "representative_results.txt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the saved model on representative labeled messages."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Training dataset used for the saved artifact, needed for custom datasets.",
    )
    args = parser.parse_args()

    model, metadata = load_model_artifact(MODEL_FILE)
    if args.dataset is not None:
        dataset = load_data(args.dataset)
    elif metadata["dataset_type"] == "custom":
        default_path = DATA_DIR / metadata["dataset_file"]
        if not default_path.is_file():
            parser.error(
                "The artifact was trained on a custom dataset outside data/. "
                "Pass its path with --dataset."
            )
        dataset = load_data(default_path)
    elif metadata["dataset_type"] == "demo":
        dataset = load_data(demo=True)
    else:
        dataset = load_data(DATA_DIR / metadata["dataset_file"])

    x_train, x_test, y_train, y_test = train_test_split(
        dataset["message"],
        dataset["label"],
        test_size=metadata["test_size"],
        random_state=metadata["random_state"],
        stratify=dataset["label"],
    )
    messages = load_data(FIXTURE_FILE)
    evaluation, matrix, _ = _score_model(
        model, messages["message"], messages["label"]
    )
    predictions = [
        label
        for label, _ in predict_messages(model, messages["message"].tolist())
    ]
    false_positives = [
        message
        for message, actual, predicted in zip(
            messages["message"], messages["label"], predictions, strict=True
        )
        if actual == "ham" and predicted == "spam"
    ]
    false_negatives = [
        message
        for message, actual, predicted in zip(
            messages["message"], messages["label"], predictions, strict=True
        )
        if actual == "spam" and predicted == "ham"
    ]
    report = classification_report(
        messages["label"],
        predictions,
        labels=list(CLASS_LABELS),
        target_names=["Ham", "Spam"],
        zero_division=0,
    )
    lines = [
        "Representative message diagnostic (small, manually labeled set)",
        f"Model artifact: {MODEL_FILE.name}",
        f"Training dataset: {metadata['dataset_name']} ({metadata['dataset_type']})",
        f"Model: {metadata['model_name']}",
        f"Diagnostic samples: {len(messages)}",
        f"Positive class: Spam",
        "",
        "Metrics on this diagnostic set (not the training hold-out set):",
        f"Accuracy: {evaluation['accuracy']:.4f}",
        f"Precision (Spam): {evaluation['precision']:.4f}",
        f"Recall (Spam): {evaluation['recall']:.4f}",
        f"F1-score (Spam): {evaluation['f1']:.4f}",
        "",
        "Confusion matrix (rows = actual [Ham, Spam], columns = predicted [Ham, Spam]):",
        str(matrix),
        "",
        "Classification report:",
        report.rstrip(),
        "",
        "Candidate comparison (all fitted on the same corpus training split):",
        "Model | CV Spam F1 (training only) | Corpus hold-out accuracy "
        "| Spam precision | Spam recall | Spam F1 | Representative accuracy "
        "| Representative Spam F1",
    ]
    for model_name, candidate in _build_models(metadata["random_state"]).items():
        candidate.set_params(
            **metadata["comparison"][model_name]["best_parameters"]
        )
        candidate.fit(x_train, y_train)
        if model_name == metadata["model_name"]:
            candidate.spam_decision_threshold_ = metadata["spam_decision"]["threshold"]
        holdout_metrics, _, _ = _score_model(candidate, x_test, y_test)
        diagnostic_metrics, _, _ = _score_model(
            candidate, messages["message"], messages["label"]
        )
        lines.append(
            f"{model_name} | "
            f"{metadata['comparison'][model_name]['cross_validation_f1']:.4f} "
            f"| {holdout_metrics['accuracy']:.4f} "
            f"| {holdout_metrics['precision']:.4f} | {holdout_metrics['recall']:.4f} "
            f"| {holdout_metrics['f1']:.4f} "
            f"| {diagnostic_metrics['accuracy']:.4f} | {diagnostic_metrics['f1']:.4f}"
        )
    lines.extend(
        [
            "",
            "This manually labeled diagnostic set is not used for cross-validation, "
            "model selection, or the corpus hold-out evaluation.",
        ]
    )
    lines.extend(
        [
            "",
            f"False positives ({len(false_positives)}):",
            *(f"- {message}" for message in false_positives),
            "",
            f"False negatives ({len(false_negatives)}):",
            *(f"- {message}" for message in false_negatives),
            "",
            "Interpretation: this small hand-labeled diagnostic set is useful for targeted "
            "checks, but is not a representative population estimate or a substitute for "
            "evaluation on the full SMS Spam Collection.",
        ]
    )
    output = "\n".join(lines) + "\n"
    REPORT_FILE.write_text(output, encoding="utf-8")
    print(output, end="")
    print(f"Saved diagnostic report to {REPORT_FILE}")


if __name__ == "__main__":
    main()
