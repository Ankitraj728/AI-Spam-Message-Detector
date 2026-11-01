import csv
import json
from pathlib import Path

import joblib
import pytest
from streamlit.testing.v1 import AppTest

from app import message_indicators
from train_model import (
    DatasetError,
    ModelArtifactError,
    load_data,
    load_model_artifact,
    _select_model_name,
    predict_message,
    train_and_evaluate,
    validate_message,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPRESENTATIVE_DATA = PROJECT_DIR / "tests" / "fixtures" / "representative_messages.csv"


def _write_dataset(path: Path, *, format_name: str = "canonical") -> Path:
    rows = []
    for index in range(40):
        rows.append(("ham", f"Meeting friend lunch tomorrow, project note {index}"))
        rows.append(("spam", f"Urgent claim cash prize click offer now {index}"))

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t" if format_name == "uci" else ",")
        if format_name == "kaggle":
            writer.writerow(["v1", "v2", "Unnamed: 2"])
            writer.writerows((label, message, "") for label, message in rows)
        elif format_name == "canonical":
            writer.writerow(["label", "message"])
            writer.writerows(rows)
        elif format_name == "uci":
            writer.writerows(rows)
    return path


@pytest.mark.parametrize(
    ("format_name", "suffix"),
    [("canonical", ".csv"), ("kaggle", ".csv"), ("uci", ".txt")],
)
def test_loads_supported_dataset_formats(
    tmp_path: Path, format_name: str, suffix: str
) -> None:
    path = _write_dataset(tmp_path / f"messages{suffix}", format_name=format_name)

    data = load_data(path)

    assert len(data) == 80
    assert set(data["label"]) == {"ham", "spam"}
    assert list(data.columns) == ["message", "label"]


def test_missing_full_dataset_is_an_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="full SMS Spam Collection dataset is missing"):
        load_data(data_dir=tmp_path)


def test_unknown_labels_and_missing_required_columns_are_rejected(tmp_path: Path) -> None:
    bad_label = tmp_path / "bad-label.csv"
    bad_label.write_text(
        "label,message\nham,Hello\nspam,Offer\npromotion,Click\n", encoding="utf-8"
    )
    with pytest.raises(DatasetError, match="unexpected labels"):
        load_data(bad_label)

    bad_columns = tmp_path / "bad-columns.csv"
    bad_columns.write_text("kind,body\nham,Hello\nspam,Offer\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="must have label and message columns"):
        load_data(bad_columns)


def test_missing_rows_and_duplicates_are_reported_and_removed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "cleaning.csv"
    path.write_text(
        "label,message\nham,Hello\nham, Hello \nspam,Offer\n,Missing label\n",
        encoding="utf-8",
    )

    data = load_data(path)
    output = capsys.readouterr().out

    assert len(data) == 2
    assert "missing labels or messages" in output
    assert "duplicate row(s)" in output


def test_deduplication_preserves_nonidentical_message_text(tmp_path: Path) -> None:
    path = tmp_path / "exact-duplicates.csv"
    path.write_text(
        "label,message\n"
        "ham,Hello there\n"
        "ham,hello there\n"
        "ham,Hello  there\n"
        "spam,Claim prize\n"
        "spam,Claim prize\n",
        encoding="utf-8",
    )

    data = load_data(path)

    assert len(data) == 4
    assert data.attrs["duplicates_removed"] == 1
    assert set(data.loc[data["label"] == "ham", "message"]) == {
        "Hello there",
        "hello there",
        "Hello  there",
    }


def test_identical_messages_with_conflicting_labels_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "conflicting.csv"
    path.write_text(
        "label,message\nham,Hello there\nspam, Hello there \nham,See you\nspam,Claim prize\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="conflicting ham/spam labels"):
        load_data(path)


def test_message_validation_rejects_empty_and_excessively_long_input() -> None:
    with pytest.raises(ValueError, match="Enter a message"):
        validate_message(" \n ")
    with pytest.raises(ValueError, match="5,000 characters or fewer"):
        validate_message("x" * 5_001)


def test_message_indicators_are_supporting_signals() -> None:
    indicators = message_indicators(
        "URGENT: verify your password at https://example.test. "
        "Call +1 555 123 4567 about your $500 refund."
    )

    assert len(indicators) == 5
    assert message_indicators("See you at the library after class.") == []


def test_malformed_csv_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "malformed.csv"
    path.write_text('label,message\nham,"Unclosed message\nspam,Click\n', encoding="utf-8")

    with pytest.raises(DatasetError, match="Could not parse dataset"):
        load_data(path)


def test_training_rejects_too_few_examples_for_stratification(tmp_path: Path) -> None:
    path = tmp_path / "small.csv"
    path.write_text(
        "label,message\nham,Meet friend\nham,See you\nspam,Claim prize\nspam,Win money\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="at least 6 usable examples"):
        train_and_evaluate(path, output_dir=tmp_path / "output")


def test_training_is_reproducible_and_model_predicts_both_classes(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path / "messages.csv")
    first = train_and_evaluate(
        dataset, output_dir=tmp_path / "first", random_state=17
    )
    second = train_and_evaluate(
        dataset, output_dir=tmp_path / "second", random_state=17
    )

    assert first["evaluation"] == second["evaluation"]
    assert first["comparison"] == second["comparison"]
    assert first["training_samples"] == second["training_samples"]
    assert first["testing_samples"] == second["testing_samples"]
    assert first["cv_folds"] == 5
    assert set(first["comparison"]) == {
        "Logistic Regression",
        "Multinomial Naive Bayes",
        "Linear SVM",
    }
    assert first["model_name"] == _select_model_name(first["comparison"])
    assert all(
        "classification_report" in result
        for result in first["comparison"].values()
    )
    assert all(
        "cross_validation_recall" in result
        and "cross_validation_f1_standard_error" in result
        for result in first["comparison"].values()
    )
    assert first["positive_class"] == "spam"
    assert first["training_samples"] + first["testing_samples"] == first["total_samples"]
    assert first["spam_probability_calibrated"] is (
        first["model_name"] == "Linear SVM"
    )
    assert (
        first["data_leakage_checks"]["train_test_message_overlap"] == 0
    )
    assert first["text_preprocessing"]["ngram_range"] == [1, 2]
    assert first["text_preprocessing"]["stop_words"] is None

    model, metadata = load_model_artifact(tmp_path / "first" / "spam_model.joblib")
    assert metadata["dataset_type"] == "custom"
    assert metadata["source_samples"] == 80
    assert metadata["duplicates_removed"] == 0
    assert predict_message(model, "urgent claim cash prize now")[0] == "spam"
    assert predict_message(model, "meeting friend lunch tomorrow")[0] == "ham"
    assert 0.0 <= metadata["spam_decision"]["threshold"] <= 1.0
    assert model.spam_decision_threshold_ == metadata["spam_decision"]["threshold"]
    assert (
        model.spam_probability_calibrated_
        is metadata["spam_probability_calibrated"]
    )
    assert (tmp_path / "first" / "results.txt").is_file()
    evaluation_json = json.loads(
        (tmp_path / "first" / "evaluation_results.json").read_text(encoding="utf-8")
    )
    assert evaluation_json["evaluation"] == first["evaluation"]
    assert (tmp_path / "first" / "confusion_matrix.png").is_file()


def test_model_selection_uses_cross_validation_not_test_metrics() -> None:
    candidates = {
        "CV winner": {
            "cross_validation_f1": 0.8,
            "metrics": {"f1": 0.2, "accuracy": 0.2, "precision": 0.2},
        },
        "Test winner": {
            "cross_validation_f1": 0.7,
            "metrics": {"f1": 1.0, "accuracy": 1.0, "precision": 1.0},
        },
    }

    assert _select_model_name(candidates) == "CV winner"


def test_model_selection_prefers_recall_within_one_standard_error() -> None:
    candidates = {
        "Higher F1": {
            "cross_validation_f1": 0.934,
            "cross_validation_f1_standard_error": 0.01,
            "cross_validation_recall": 0.88,
        },
        "Higher recall": {
            "cross_validation_f1": 0.933,
            "cross_validation_f1_standard_error": 0.01,
            "cross_validation_recall": 0.94,
        },
        "Outside one standard error": {
            "cross_validation_f1": 0.90,
            "cross_validation_f1_standard_error": 0.01,
            "cross_validation_recall": 1.0,
        },
    }

    assert _select_model_name(candidates) == "Higher recall"


def test_complete_uci_dataset_is_available_and_deduplicated() -> None:
    data = load_data(PROJECT_DIR / "data" / "SMSSpamCollection")

    assert len(data) > 5_000
    assert len(data) == 5_160
    assert data.attrs["source_samples"] == 5_574
    assert data.attrs["source_class_counts"] == {"ham": 4_827, "spam": 747}
    assert data.attrs["duplicates_removed"] == 414
    assert data.attrs["source_samples"] == (
        len(data) + data.attrs["duplicates_removed"]
    )
    assert data.attrs["duplicates_removed"] > 0
    assert set(data["label"]) == {"ham", "spam"}
    assert not data["message"].duplicated().any()


def test_saved_evaluation_artifacts_use_the_full_corpus() -> None:
    metadata = json.loads(
        (PROJECT_DIR / "evaluation_results.json").read_text(encoding="utf-8")
    )
    model, artifact_metadata = load_model_artifact(PROJECT_DIR / "spam_model.joblib")

    assert metadata == artifact_metadata
    assert metadata["dataset_type"] == "full"
    assert metadata["dataset_file"] == "SMSSpamCollection"
    assert metadata["source_samples"] == 5_574
    assert metadata["total_samples"] == 5_160
    assert metadata["training_samples"] == 4_128
    assert metadata["testing_samples"] == 1_032
    assert metadata["data_leakage_checks"]["train_test_message_overlap"] == 0
    assert metadata["data_leakage_checks"][
        "threshold_uses_nested_training_only_cross_validation"
    ]
    assert model.spam_probability_calibrated_
    assert metadata["evaluation"]["confusion_matrix"] == metadata["comparison"][
        metadata["model_name"]
    ]["confusion_matrix"]
    assert (PROJECT_DIR / "results.txt").is_file()
    assert (PROJECT_DIR / "confusion_matrix.png").is_file()


def test_representative_messages_are_unseen_and_metrics_come_from_model(
    tmp_path: Path,
) -> None:
    dataset = _write_dataset(tmp_path / "messages.csv")
    train_and_evaluate(dataset, output_dir=tmp_path / "trained")
    model, _ = load_model_artifact(tmp_path / "trained" / "spam_model.joblib")
    representative = load_data(REPRESENTATIVE_DATA)
    sample_training = load_data(demo=True)
    normalize = lambda message: " ".join(message.casefold().split())

    training_messages = {normalize(message) for message in sample_training["message"]}
    assert not any(
        normalize(message) in training_messages for message in representative["message"]
    )

    predictions = model.predict(representative["message"])
    assert len(predictions) == 20
    assert set(predictions).issubset({"ham", "spam"})


def test_missing_and_outdated_model_artifacts_are_reported(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactError, match="Model artifact not found"):
        load_model_artifact(tmp_path / "missing.joblib")

    outdated_path = tmp_path / "outdated.joblib"
    joblib.dump({"model": "legacy"}, outdated_path)
    with pytest.raises(ModelArtifactError, match="outdated or invalid"):
        load_model_artifact(outdated_path)


def test_streamlit_application_starts_without_uncaught_errors() -> None:
    app = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=20).run()

    assert not app.exception
    assert any("AI Spam Message Detector" in title.value for title in app.title)
    theme_styles = next(
        element.value for element in app.markdown if "<style>" in element.value
    )
    assert "var(--background-color)" in theme_styles
    assert "var(--secondary-background-color)" in theme_styles
    assert "var(--text-color)" in theme_styles
    assert [button.label for button in app.button[:4]] == [
        "Load Spam Example",
        "Load Legitimate Example",
        "Clear / Reset",
        "Analyze Message",
    ]
    assert "prediction" not in app.session_state
    assert {
        "Project Overview",
        "Model Performance",
        "Model Comparison",
        "Dataset Analysis",
        "Model Information and Methodology",
        "Limitations and Future Improvements",
    }.issubset({expander.label for expander in app.expander})

    app.text_area[0].input(
        "Your SIM will be deactivated. Update your KYC now using this link."
    ).run()
    assert "prediction" not in app.session_state
    next(button for button in app.button if button.label == "Analyze Message").click().run()
    assert app.session_state["prediction"]["label"] == "spam"
    assert app.session_state["prediction"]["spam_probability"] is not None
    probability_text = f"{app.session_state['prediction']['spam_probability']:.1%}"
    assert any(
        probability_text in element.value and "10.5% Spam cutoff" in element.value
        for element in app.markdown
    )
    assert any(
        f"{probability_text} is the classifier's estimate" in element.value
        for element in app.caption
    )

    app.text_area[0].input(
        "LookAtMe!: Thanks for your purchase of a video clip from LookAtMe!, "
        "you've been charged 35p. Think you can do better? Why not send a video "
        "in a MMSto 32323."
    ).run()
    next(button for button in app.button if button.label == "Analyze Message").click().run()
    assert app.session_state["prediction"]["label"] == "spam"

    for amount in ("5000", "50000"):
        app.text_area[0].input(
            f"Earn {amount} per day working from home. No experience needed. "
            "Register today."
        ).run()
        next(
            button for button in app.button if button.label == "Analyze Message"
        ).click().run()
        assert app.session_state["prediction"]["label"] == "spam"
        probability_text = f"{app.session_state['prediction']['spam_probability']:.1%}"
        assert any(
            probability_text in element.value and "10.5% Spam cutoff" in element.value
            for element in app.markdown
        )
