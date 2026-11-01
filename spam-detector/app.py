"""Streamlit interface for the locally running SMS spam message detector."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from train_model import (
    DATA_DIR,
    MAX_MESSAGE_LENGTH,
    DatasetError,
    MODEL_FILE,
    ModelArtifactError,
    load_data,
    load_model_artifact,
    predict_message,
    predict_messages,
)

SPAM_EXAMPLES = [
    "Congratulations! You have won a free prize. Call now to claim your reward.",
    "URGENT! Your bank account is blocked. Click the link to verify immediately.",
    "WINNER!! You have been selected for a 1000 cash prize. Reply YES to claim.",
    "Free entry in a weekly competition to win a brand new phone. Text WIN to 80085.",
    "Your SIM will be deactivated. Update your KYC now using this link.",
    "Earn 5000 per day working from home. No experience needed. Register today.",
]
HAM_EXAMPLES = [
    "Are we meeting for lunch today?",
    "Please send me the assignment notes when you get time.",
    "I will reach home by 8 pm. Save some dinner for me.",
    "The lab session is moved to Thursday at 10 am.",
    "Don't forget to submit the project report tomorrow.",
    "Can you share the notes from today's lecture?",
]
URGENCY_PATTERN = re.compile(
    r"\b(?:urgent|immediately|act now|hurry|last chance|final notice|expires today)\b",
    re.IGNORECASE,
)
SENSITIVE_INFO_PATTERN = re.compile(
    r"\b(?:password|passcode|bank details|account number|card details|"
    r"social security|ssn|verify your account|financial information)\b",
    re.IGNORECASE,
)
LINK_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)")
MONEY_PATTERN = re.compile(
    r"(?:[$€£]\s?\d|\b\d+(?:[.,]\d+)?\s?(?:usd|dollars?|euros?|pounds?)\b)",
    re.IGNORECASE,
)
MONEY_WORD_PATTERN = re.compile(
    r"\b(?:cash|prize|refund|loan|payment|credit card|money|bank)\b", re.IGNORECASE
)


@st.cache_resource
def _cached_model(
    model_path: str, modified_ns: int, size_bytes: int
) -> tuple[Pipeline, dict[str, Any]]:
    return load_model_artifact(model_path)


@st.cache_data
def _cached_dataset(file_path: str, modified_ns: int, size_bytes: int) -> pd.DataFrame:
    return load_data(file_path)


def message_indicators(message: str) -> list[str]:
    """Return possible message signals; these are not proof that a message is spam."""
    indicators = []
    if URGENCY_PATTERN.search(message):
        indicators.append("Urgency or pressure language")
    if SENSITIVE_INFO_PATTERN.search(message):
        indicators.append("Request for sensitive or financial information")
    if LINK_PATTERN.search(message):
        indicators.append("Contains a web link")
    if PHONE_PATTERN.search(message):
        indicators.append("Contains a phone-number-like sequence")
    if MONEY_PATTERN.search(message) or MONEY_WORD_PATTERN.search(message):
        indicators.append("Mentions money, prizes, or financial activity")
    return indicators


def _clear_result() -> None:
    st.session_state.pop("prediction", None)


def _load_example(example: str) -> None:
    st.session_state["message"] = example
    _clear_result()


def _clear_message() -> None:
    st.session_state["message"] = ""
    _clear_result()


def _show_performance(metadata: dict[str, Any]) -> None:
    evaluation = metadata["evaluation"]
    st.subheader(f"Held-out test performance · {metadata['model_name']}")
    st.caption(
        f"Stratified split: {metadata['training_samples']:,} training messages and "
        f"{metadata['testing_samples']:,} held-out test messages "
        f"({metadata['test_size']:.0%} test); Spam is the positive class. "
        "Cross-validation scores are reported separately in Model Comparison."
    )
    st.info(
        f"The selected model uses a {metadata['spam_decision']['threshold']:.1%} "
        f"{metadata['spam_decision']['threshold_basis']} threshold, tuned using "
        "training data only. This recall-focused setting can flag legitimate messages "
        "as spam; review flagged messages before acting on them."
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Accuracy", f"{evaluation['accuracy']:.1%}")
    metric_columns[1].metric("Precision · Spam", f"{evaluation['precision']:.1%}")
    metric_columns[2].metric("Recall · Spam", f"{evaluation['recall']:.1%}")
    metric_columns[3].metric("F1-score · Spam", f"{evaluation['f1']:.1%}")

    st.markdown("**Confusion matrix**")
    matrix = evaluation["confusion_matrix"]
    st.dataframe(
        pd.DataFrame(
            matrix,
            index=["Actual Ham", "Actual Spam"],
            columns=["Predicted Ham", "Predicted Spam"],
        ),
        width="stretch",
    )
    st.caption(
        "Rows are actual labels; columns are predicted labels. "
        "Metrics are calculated from saved predictions on the held-out test split. "
        f"False positives (legitimate labeled Spam): {matrix[0][1]}. "
        f"False negatives (Spam labeled legitimate): {matrix[1][0]}."
    )
    st.caption(
        "A false positive can cause a legitimate message to be overlooked; "
        "a false negative means a spam message was not caught."
    )


def _show_comparison(metadata: dict[str, Any]) -> None:
    comparison = metadata.get("comparison", {})
    if comparison:
        st.subheader("Model comparison")
        st.caption(
            "Selection uses training-only cross-validation: highest Spam F1 defines "
            "the best score, then recall breaks ties within one standard error. "
            "Test metrics are reported for comparison and never used for selection."
        )
        st.caption(
            "The table separates cross-validation metrics on training folds from "
            "metrics on the held-out test set. In-sample training scores are not "
            "shown because they can overstate generalization."
        )
        st.caption(
            "The selected model's test metrics use a decision threshold tuned on "
            "training-only out-of-fold scores. Other models use their default "
            "prediction threshold. CV metrics are not test metrics."
        )
        comparison_rows = [
            {
                "Model": name,
                "CV F1 · Spam (train only)": result["cross_validation_f1"],
                "CV recall · Spam (train only)": result["cross_validation_recall"],
                "Spam cutoff": (
                    f"{result['spam_decision_threshold']:.1%}"
                    if "spam_decision_threshold" in result
                    else "Default"
                ),
                "Accuracy": result["metrics"]["accuracy"],
                "Precision · Spam": result["metrics"]["precision"],
                "Recall · Spam": result["metrics"]["recall"],
                "F1 · Spam": result["metrics"]["f1"],
            }
            for name, result in comparison.items()
        ]
        cv_scores = pd.DataFrame(
            {
                "Mean CV Spam F1": {
                    name: result["cross_validation_f1"]
                    for name, result in comparison.items()
                }
            }
        )
        st.bar_chart(cv_scores, y="Mean CV Spam F1", color="#3B82F6")
        st.dataframe(
            pd.DataFrame(comparison_rows)
            .set_index("Model")
            .style.format(
                {
                    "CV F1 · Spam (train only)": "{:.1%}",
                    "CV recall · Spam (train only)": "{:.1%}",
                    "Accuracy": "{:.1%}",
                    "Precision · Spam": "{:.1%}",
                    "Recall · Spam": "{:.1%}",
                    "F1 · Spam": "{:.1%}",
                }
            ),
            width="stretch",
        )
        with st.expander("Classification reports on held-out test data"):
            for name, result in comparison.items():
                st.markdown(f"**{name}**")
                st.code(result["classification_report"], language=None)


def _common_words(messages: pd.Series) -> pd.DataFrame:
    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=8_000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    counts = vectorizer.fit_transform(messages)
    totals = counts.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()
    order = totals.argsort()[::-1][:10]
    return pd.DataFrame(
        {"Word": words[order], "Occurrences": totals[order]}
    ).set_index("Word")


def _show_dataset_analysis(data: pd.DataFrame, metadata: dict[str, Any]) -> None:
    st.subheader("Dataset analysis")
    st.caption(
        "Descriptive statistics for the included corpus; this analysis is not used "
        "to fit or select the classifier."
    )
    class_columns = st.columns(2)
    class_counts = metadata["class_counts"]
    class_columns[0].metric("Ham messages", f"{class_counts['ham']:,}")
    class_columns[1].metric("Spam messages", f"{class_counts['spam']:,}")
    distribution = pd.DataFrame(
        {"Messages": [class_counts["ham"], class_counts["spam"]]},
        index=["Ham", "Spam"],
    )
    st.markdown("**Spam versus ham distribution**")
    st.bar_chart(distribution, y="Messages", color="#10B981")

    lengths = data.assign(
        message_length=data["message"].str.len(),
        length_band=pd.cut(
            data["message"].str.len(),
            bins=[0, 50, 100, 160, 320, 1_000, float("inf")],
            labels=["1–50", "51–100", "101–160", "161–320", "321–1,000", "1,000+"],
            include_lowest=True,
        ),
    )
    length_counts = (
        lengths.groupby(["length_band", "label"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["ham", "spam"], fill_value=0)
    )
    st.markdown("**Message length distribution (characters)**")
    st.bar_chart(length_counts, y=["ham", "spam"], color=["#2563EB", "#F97316"])
    st.caption(
        f"Median message length: {data['message'].str.len().median():.0f} characters."
    )

    word_columns = st.columns(2)
    for column, label, title in (
        (word_columns[0], "spam", "Common spam words"),
        (word_columns[1], "ham", "Common ham words"),
    ):
        column.markdown(f"**{title}**")
        try:
            terms = _common_words(data.loc[data["label"] == label, "message"])
            column.bar_chart(
                terms,
                y="Occurrences",
                color="#F97316" if label == "spam" else "#2563EB",
            )
        except ValueError as error:
            column.info(f"Word frequencies are unavailable: {error}")


def _show_overview(metadata: dict[str, Any]) -> None:
    st.subheader("Dataset information")
    st.write(
        "This project classifies SMS text as likely Spam or Legitimate using "
        "Logistic Regression, Multinomial Naive Bayes, and Linear SVM."
    )
    source = metadata["dataset_source"]
    source_display = (
        f"[UCI Machine Learning Repository]({source})"
        if source.startswith("https://")
        else source
    )
    st.markdown(
        f"**Dataset:** {metadata['dataset_name']}  \n"
        f"**Source:** {source_display}  \n"
        f"**Dataset size:** {metadata['total_samples']:,} unique messages after removing "
        f"{metadata['duplicates_removed']:,} exact duplicates "
        f"from {metadata['source_samples']:,} source rows."
    )
    if metadata["dataset_type"] == "full":
        st.caption(
            "Original source: SMS Spam Collection, UCI Machine Learning Repository "
            "(public dataset record linked above). Exact duplicates are removed "
            "before the stratified split so identical messages cannot occur in both "
            "partitions."
        )
    elif metadata["dataset_type"] == "demo":
        st.warning(
            "This artifact was trained on the small bundled demo dataset. Its metrics "
            "are not final corpus evaluation results. Run `python train_model.py` "
            "without `--demo` to train on the full SMS Spam Collection."
        )
    else:
        st.caption(
            "This is a user-specified dataset. Confirm its provenance before "
            "presenting results as a corpus evaluation."
        )
    overview = st.columns(4)
    overview[0].metric("Unique messages", f"{metadata['total_samples']:,}")
    overview[1].metric("Ham", f"{metadata['class_counts']['ham']:,}")
    overview[2].metric("Spam", f"{metadata['class_counts']['spam']:,}")
    overview[3].metric("Duplicates removed", f"{metadata['duplicates_removed']:,}")
    st.markdown("**Class distribution**")
    st.bar_chart(
        pd.DataFrame(
            {
                "Messages": [
                    metadata["class_counts"]["ham"],
                    metadata["class_counts"]["spam"],
                ]
            },
            index=["Ham", "Spam"],
        ),
        y="Messages",
        color="#3B82F6",
    )


def _show_model_info(metadata: dict[str, Any]) -> None:
    st.subheader("Model and evaluation methodology")
    st.markdown(
        """
        - **TF-IDF:** word unigrams/bigrams and character n-grams capture both message
          meaning and spelling/formatting patterns. Each vectorizer is inside the
          scikit-learn pipeline and is fitted independently inside every training fold.
        - **Logistic Regression:** a linear classifier that estimates class probabilities
          from weighted text features.
        - **Multinomial Naive Bayes:** a probabilistic classifier that models token
          occurrence counts and works effectively with sparse text features.
        - **Linear SVM:** a maximum-margin linear classifier. Its scores are sigmoid
          calibrated with training-fold data. Probability estimates are displayed only
          for this calibrated model; scores from the other classifiers are not presented
          as confidence.
        """
    )
    st.markdown(
        f"**Method:** exact duplicate removal before splitting; stratified "
        f"{1 - metadata['test_size']:.0%}/{metadata['test_size']:.0%} "
        f"train/test split (seed {metadata['random_state']}); "
        f"{metadata['cv_folds']}-fold stratified cross-validation on training data only. "
        "Highest Spam F1 defines the best score; candidates within one standard error "
        "are treated as comparable, then higher training-only Spam recall is preferred. "
        "The held-out test split is used only for the final metrics and reports. "
        "TF-IDF fitting is inside each model pipeline and is repeated within CV folds."
    )
    spam_decision = metadata["spam_decision"]
    st.caption(
        f"The selected model's {spam_decision['threshold']:.1%} "
        f"{spam_decision['threshold_basis']} cutoff "
        f"was chosen from training-only out-of-fold predictions to target "
        f"{spam_decision['target_recall']:.0%} Spam recall. A lower cutoff can catch "
        "more spam but will also flag more legitimate messages."
    )
    if metadata["spam_probability_calibrated"]:
        st.caption(
            f"Displayed probability estimates use {metadata['probability_calibration']}. "
            "They are estimates, not guarantees of correctness."
        )
    else:
        st.caption(
            "The selected model does not provide calibrated probabilities, so the "
            "application does not display prediction confidence."
        )
    checks = metadata["data_leakage_checks"]
    st.markdown(
        "**Leakage checks:** TF-IDF inside pipelines: "
        f"{'passed' if checks['tfidf_is_inside_pipeline'] else 'failed'}; "
        "train/test message overlap: "
        f"{checks['train_test_message_overlap']}; "
        "nested threshold cross-validation: "
        f"{'passed' if checks['threshold_uses_nested_training_only_cross_validation'] else 'failed'}; "
        f"test data used for selection: {'yes' if checks['test_data_used_for_model_selection'] else 'no'}."
    )


def _show_detection(model: Pipeline) -> None:
    if "message" not in st.session_state:
        st.session_state["message"] = ""

    with st.container(border=True, key="detector-card"):
        st.text_area(
            "Message text",
            key="message",
            height=160,
            placeholder="Paste or type the message you want to check…",
            on_change=_clear_result,
            label_visibility="collapsed",
            max_chars=MAX_MESSAGE_LENGTH,
        )
        st.caption(
            f"Up to {MAX_MESSAGE_LENGTH:,} characters · "
            "No external classification API is used."
        )

        example_columns = st.columns(3, gap="small")
        example_columns[0].button(
            "Load Spam Example",
            on_click=_load_example,
            args=(random.choice(SPAM_EXAMPLES),),
            width="stretch",
            key="load_spam_example",
        )
        example_columns[1].button(
            "Load Legitimate Example",
            on_click=_load_example,
            args=(random.choice(HAM_EXAMPLES),),
            width="stretch",
            key="load_ham_example",
        )
        example_columns[2].button(
            "Clear / Reset",
            on_click=_clear_message,
            width="stretch",
            key="clear_message",
        )

        if st.button(
            "Analyze Message",
            type="primary",
            width="stretch",
            key="analyze_single",
        ):
            try:
                label, spam_probability = predict_message(
                    model, st.session_state.get("message", "")
                )
                st.session_state["prediction"] = {
                    "label": label,
                    "spam_probability": spam_probability,
                }
            except (ValueError, TypeError) as error:
                _clear_result()
                st.error(str(error))

    prediction = st.session_state.get("prediction")
    if prediction:
        spam = prediction["label"] == "spam"
        result_title = "Likely Spam" if spam else "Likely Legitimate"
        result_color = "#b42318" if spam else "#16803c"
        result_icon = "!" if spam else "✓"
        st.markdown(
            f"""
            <div class="result-card" style="--result-color:{result_color}">
              <span class="result-icon">{result_icon}</span>
              <div>
                <div class="result-eyebrow">ANALYSIS RESULT</div>
                <div class="result-title">{result_title}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        spam_probability = prediction.get("spam_probability")
        if spam_probability is not None:
            threshold = model.spam_decision_threshold_
            decision_text = (
                f"The Spam estimate ({spam_probability:.1%}) is at or above the "
                f"{threshold:.1%} Spam cutoff."
                if spam
                else f"The Spam estimate ({spam_probability:.1%}) is below the "
                f"{threshold:.1%} Spam cutoff."
            )
            st.markdown(
                f'<div class="decision-note"><strong>Why this label?</strong> '
                f'{decision_text}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("**Model-estimated spam probability**")
            st.progress(spam_probability)
            st.caption(
                f"{spam_probability:.1%} is the classifier's estimate, not a "
                "guaranteed real-world probability. The cutoff controls the Spam/Ham "
                "label; it does not change this estimate."
            )
        else:
            st.caption(
                "Confidence is not displayed because this model's scores have not "
                "been calibrated as probabilities."
            )
        st.warning(
            "Predictions are not always correct. Check the original message and "
            "sender before taking action."
        )

        with st.expander("Optional message indicators"):
            st.caption(
                "These simple text signals can support your review. They do not "
                "determine the model prediction."
            )
            indicators = message_indicators(st.session_state.get("message", ""))
            if indicators:
                for indicator in indicators:
                    st.write(f"• {indicator}")
            else:
                st.write("No listed indicators were detected.")


def _show_batch_detection(model: Pipeline) -> None:
    st.caption(
        "Enter one message per line. Each message is classified independently. "
        f"The recall-focused decision threshold is "
        f"{model.spam_decision_threshold_:.1%}; this can increase false positives. "
        "Confidence is shown only when the selected model is calibrated."
    )
    st.text_area(
        "Messages",
        key="batch_messages",
        height=160,
        placeholder="Message one\nMessage two\nMessage three",
    )
    if st.button("Analyze Messages", type="primary", key="analyze_batch"):
        messages = [
            message.strip()
            for message in st.session_state.get("batch_messages", "").splitlines()
            if message.strip()
        ]
        try:
            results = predict_messages(model, messages)
            st.session_state["batch_predictions"] = [
                {
                    "Message": message,
                    "Prediction": "Spam" if label == "spam" else "Ham",
                    "Spam probability": (
                        f"{probability:.1%}" if probability is not None else "Unavailable"
                    ),
                }
                for message, (label, probability) in zip(messages, results, strict=True)
            ]
        except (ValueError, TypeError) as error:
            st.session_state.pop("batch_predictions", None)
            st.error(str(error))
    if st.session_state.get("batch_predictions"):
        st.dataframe(
            pd.DataFrame(st.session_state["batch_predictions"]),
            width="stretch",
            hide_index=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="AI Spam Message Detector",
        page_icon="📩",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp { background: var(--background-color); }
        .block-container {
            max-width: 820px;
            padding: 2.5rem 1.5rem 3.5rem;
        }
        h1 {
            color: var(--text-color);
            font-size: clamp(2rem, 5vw, 2.8rem);
            letter-spacing: -0.045em;
            line-height: 1.1;
            text-align: center;
        }
        .app-subtitle {
            color: var(--text-color);
            font-size: 1.05rem;
            margin: 0 auto 1.5rem;
            max-width: 38rem;
            opacity: 0.72;
            text-align: center;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--secondary-background-color);
            border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
            border-radius: 18px;
            box-shadow: 0 8px 28px rgba(16, 24, 40, 0.045);
        }
        [data-testid="stTextArea"] textarea {
            background: var(--secondary-background-color);
            border: 1px solid color-mix(in srgb, var(--text-color) 18%, transparent);
            border-radius: 12px;
            color: var(--text-color);
            line-height: 1.6;
        }
        [data-testid="stTextArea"] textarea:focus {
            border-color: #d92d20;
            box-shadow: 0 0 0 2px rgba(217, 45, 32, 0.12);
        }
        [data-testid="stMetric"] {
            background: var(--secondary-background-color);
            border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
            border-radius: 12px;
            padding: 0.8rem 1rem;
        }
        div.stButton > button {
            min-height: 2.8rem;
            border-radius: 10px;
            font-weight: 600;
        }
        div.stButton > button:hover {
            box-shadow: 0 4px 12px rgba(16, 24, 40, 0.1);
        }
        div.stButton > button[kind="primary"] {
            background: #d92d20;
            border-color: #d92d20;
            color: #ffffff;
            min-height: 3.2rem;
            font-size: 1rem;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #b42318;
            border-color: #b42318;
        }
        .result-card {
            align-items: center;
            background: color-mix(in srgb, var(--result-color) 10%, var(--background-color));
            border: 1px solid color-mix(in srgb, var(--result-color) 24%, transparent);
            border-radius: 14px;
            color: color-mix(in srgb, var(--result-color) 65%, var(--text-color));
            display: flex;
            gap: 0.9rem;
            margin: 1.5rem 0 1rem;
            padding: 1rem 1.15rem;
        }
        .result-icon {
            align-items: center;
            border: 2px solid currentColor;
            border-radius: 50%;
            display: inline-flex;
            flex: 0 0 2.2rem;
            font-size: 1.15rem;
            font-weight: 700;
            height: 2.2rem;
            justify-content: center;
            width: 2.2rem;
        }
        .result-eyebrow {
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            opacity: 0.8;
        }
        .result-title {
            font-size: 1.3rem;
            font-weight: 700;
        }
        .decision-note {
            background: color-mix(in srgb, #d92d20 8%, var(--background-color));
            border: 1px solid color-mix(in srgb, #d92d20 24%, transparent);
            border-left: 4px solid #d92d20;
            border-radius: 10px;
            color: color-mix(in srgb, #d92d20 65%, var(--text-color));
            line-height: 1.5;
            margin: 0.75rem 0 1rem;
            padding: 0.75rem 0.9rem;
        }
        [data-testid="stExpander"] {
            background: var(--secondary-background-color);
            border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
            border-radius: 12px;
            margin-top: 0.65rem;
        }
        @media (max-width: 640px) {
            .block-container { padding: 1.5rem 1rem 2.5rem; }
            [data-testid="column"] { min-width: 100% !important; }
            .result-title { font-size: 1.15rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("AI Spam Message Detector")
    st.markdown(
        '<p class="app-subtitle">A simple, thoughtful way to check whether a text '
        "message may be spam.</p>",
        unsafe_allow_html=True,
    )

    try:
        artifact_path = Path(MODEL_FILE)
        if artifact_path.is_file():
            artifact_stat = artifact_path.stat()
            model, metadata = _cached_model(
                str(artifact_path), artifact_stat.st_mtime_ns, artifact_stat.st_size
            )
        else:
            model, metadata = load_model_artifact(artifact_path)
    except ModelArtifactError as error:
        st.error(str(error))
        st.info(
            "Run `python train_model.py` from the project directory to generate the "
            "model artifact and evaluation results."
        )
        st.stop()

    _show_detection(model)

    with st.expander("Try multiple messages"):
        _show_batch_detection(model)

    with st.expander("Project Overview"):
        _show_overview(metadata)

    with st.expander("Model Performance"):
        _show_performance(metadata)

    with st.expander("Model Comparison"):
        _show_comparison(metadata)

    with st.expander("Dataset Analysis"):
        dataset_path = DATA_DIR / metadata["dataset_file"]
        if metadata["dataset_type"] in {"full", "demo"} and dataset_path.is_file():
            try:
                dataset_stat = dataset_path.stat()
                analysis_data = _cached_dataset(
                    str(dataset_path), dataset_stat.st_mtime_ns, dataset_stat.st_size
                )
                _show_dataset_analysis(analysis_data, metadata)
            except DatasetError as error:
                st.error(f"Could not load dataset analysis: {error}")
        else:
            st.warning(
                "Dataset analysis requires the training dataset file in the project's "
                "`data/` directory."
            )

    with st.expander("Model Information and Methodology"):
        _show_model_info(metadata)

    with st.expander("Limitations and Future Improvements"):
        st.markdown(
            """
            **Limitations**
            - Predictions can be wrong. A false positive may hide a legitimate SMS;
              a false negative may let spam through. Do not treat the output as proof.
            - The model reflects an older English-language SMS corpus and may not
              generalize to modern scams, other languages, or other message channels.
            - A random stratified test split estimates performance on similar data;
              it does not measure future performance or protect against dataset shift.
            - A calibrated score is still an estimate and depends on representative
              calibration data.

            **Future improvements**
            - Evaluate on a separately sourced, newer, time-based dataset.
            - Measure calibration and performance across language, message type, and
              false-positive/false-negative costs.
            - Collect reviewed errors with consent and retrain only with documented,
              representative labels.
            """
        )


if __name__ == "__main__":
    main()
