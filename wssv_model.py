"""Data loading, model training, evaluation and explanation for the WSSV symptom checker.

Kept separate from the Streamlit UI so it can be reused (and later extended to
several diseases) without touching the website code.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = Path(__file__).parent / "data" / "wssv_symptoms.csv"
RANDOM_STATE = 42

# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

# Rows in the supplied file that are not usable symptom descriptions.
# Keyed by the original row id (first column of the CSV).
KNOWN_BAD_ROWS = {
    5: "Not a symptom description (looks like stray text), but labelled as WSSV",
}


@dataclass
class Dataset:
    clean: pd.DataFrame  # columns: row_id, text, label (1 = WSSV, 0 = not WSSV)
    removed: pd.DataFrame  # columns: row_id, text, label, reason
    raw_count: int


def _read_csv(path: Path) -> pd.DataFrame:
    # The file was saved on Windows (cp1252); fall back to utf-8 if that changes.
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path.name}")


def normalise_text(text: str) -> str:
    text = str(text).replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(path: Path = DATA_PATH) -> Dataset:
    raw = _read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    id_col = raw.columns[0]

    frame = pd.DataFrame(
        {
            "row_id": raw[id_col].astype(int),
            "text": raw["message"].map(normalise_text),
            "label": raw["category"].astype(str).str.strip().str.lower().eq("yes").astype(int),
        }
    )

    removed_rows = []
    keep = pd.Series(True, index=frame.index)

    for idx, row in frame.iterrows():
        if row["row_id"] in KNOWN_BAD_ROWS:
            removed_rows.append({**row, "reason": KNOWN_BAD_ROWS[row["row_id"]]})
            keep[idx] = False
        elif not row["text"]:
            removed_rows.append({**row, "reason": "Empty text"})
            keep[idx] = False

    dedupe_key = frame["text"].str.lower().str.rstrip(". ")
    seen: dict[str, int] = {}
    for idx in frame.index:
        if not keep[idx]:
            continue
        key = dedupe_key[idx]
        if key in seen:
            row = frame.loc[idx]
            removed_rows.append({**row, "reason": f"Duplicate of row {seen[key]}"})
            keep[idx] = False
        else:
            seen[key] = int(frame.loc[idx, "row_id"])

    removed = pd.DataFrame(removed_rows, columns=["row_id", "text", "label", "reason"])
    return Dataset(clean=frame[keep].reset_index(drop=True), removed=removed, raw_count=len(frame))


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


def _word_and_char_features() -> FeatureUnion:
    return FeatureUnion(
        [
            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ]
    )


MODEL_INFO = {
    "Linear SVM": "Support Vector Machine on word and character patterns. Strong on small text datasets.",
    "Logistic Regression": "A simple, transparent model that weighs each word pattern for or against WSSV.",
    "Naive Bayes": "A classic probabilistic text model based on how often words occur in each class.",
    "Random Forest": "Many decision trees voting together on word and character patterns.",
}


def build_models() -> dict[str, Pipeline]:
    return {
        "Linear SVM": Pipeline(
            [
                ("features", _word_and_char_features()),
                ("model", CalibratedClassifierCV(LinearSVC(C=1.0), cv=3)),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("features", _word_and_char_features()),
                ("model", LogisticRegression(C=10, max_iter=2000)),
            ]
        ),
        "Naive Bayes": Pipeline(
            [
                ("features", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
                ("model", ComplementNB(alpha=0.3)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("features", _word_and_char_features()),
                ("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
            ]
        ),
    }


@dataclass
class ModelReport:
    name: str
    metrics: dict[str, float]  # mean over repeated cross-validation
    metrics_std: dict[str, float]
    confusion: np.ndarray  # from out-of-fold predictions, rows = actual [not, wssv]
    roc_fpr: np.ndarray
    roc_tpr: np.ndarray
    oof_proba: np.ndarray


@dataclass
class TrainedSystem:
    models: dict[str, Pipeline]
    reports: dict[str, ModelReport]
    best_model: str
    cv_description: str
    misclassified: dict[str, pd.DataFrame] = field(default_factory=dict)


METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1 score", "ROC AUC"]


def _scores(y_true, proba) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1 score": f1_score(y_true, pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, proba) if len(set(y_true)) > 1 else float("nan"),
    }


def train_and_evaluate(dataset: Dataset, n_splits: int = 5, n_repeats: int = 10) -> TrainedSystem:
    texts = dataset.clean["text"].to_numpy()
    labels = dataset.clean["label"].to_numpy()

    repeated = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    splits = list(repeated.split(texts, labels))

    reports: dict[str, ModelReport] = {}
    fitted: dict[str, Pipeline] = {}
    misclassified: dict[str, pd.DataFrame] = {}

    for name, pipeline in build_models().items():
        fold_scores = []
        for train_idx, test_idx in splits:
            model = build_models()[name]
            model.fit(texts[train_idx], labels[train_idx])
            proba = model.predict_proba(texts[test_idx])[:, 1]
            fold_scores.append(_scores(labels[test_idx], proba))

        score_frame = pd.DataFrame(fold_scores)

        oof = cross_val_predict(
            build_models()[name],
            texts,
            labels,
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE),
            method="predict_proba",
        )[:, 1]
        fpr, tpr, _ = roc_curve(labels, oof)

        reports[name] = ModelReport(
            name=name,
            metrics=score_frame.mean().to_dict(),
            metrics_std=score_frame.std().to_dict(),
            confusion=confusion_matrix(labels, (oof >= 0.5).astype(int), labels=[0, 1]),
            roc_fpr=fpr,
            roc_tpr=tpr,
            oof_proba=oof,
        )

        wrong = (oof >= 0.5).astype(int) != labels
        misclassified[name] = pd.DataFrame(
            {
                "Row": dataset.clean["row_id"][wrong].to_numpy(),
                "Text": texts[wrong],
                "Actual": np.where(labels[wrong] == 1, "WSSV", "Not WSSV"),
                "WSSV probability": oof[wrong].round(3),
            }
        )

        pipeline.fit(texts, labels)
        fitted[name] = pipeline

    best = max(reports, key=lambda n: (reports[n].metrics["F1 score"], reports[n].metrics["ROC AUC"]))
    return TrainedSystem(
        models=fitted,
        reports=reports,
        best_model=best,
        cv_description=f"{n_splits}-fold cross-validation repeated {n_repeats} times ({n_splits * n_repeats} train/test splits)",
        misclassified=misclassified,
    )


def predict_proba(model: Pipeline, texts: list[str]) -> np.ndarray:
    return model.predict_proba([normalise_text(t) for t in texts])[:, 1]


# --------------------------------------------------------------------------------------
# Explanation: which words moved the prediction
# --------------------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+|[^\sA-Za-z\d]")


def word_influence(model: Pipeline, text: str) -> list[tuple[str, float | None]]:
    """Leave-one-word-out influence for each token.

    Returns (token, influence) pairs in order. Influence is the drop in WSSV
    probability when the word is removed: positive = pushed towards WSSV,
    negative = pushed towards not WSSV. Punctuation gets None.
    """
    text = normalise_text(text)
    tokens = TOKEN_RE.findall(text)
    word_positions = [i for i, tok in enumerate(tokens) if tok[0].isalnum()]
    if not word_positions:
        return [(tok, None) for tok in tokens]

    variants = [" ".join(tok for j, tok in enumerate(tokens) if j != i) for i in word_positions]
    probs = model.predict_proba([" ".join(tokens)] + variants)[:, 1]
    base = probs[0]

    influence: dict[int, float] = {pos: float(base - p) for pos, p in zip(word_positions, probs[1:])}
    return [(tok, influence.get(i)) for i, tok in enumerate(tokens)]


# --------------------------------------------------------------------------------------
# Symptom evidence: domain knowledge check, independent of the ML model
# --------------------------------------------------------------------------------------

SYMPTOMS: list[tuple[str, str, str]] = [
    # (key, label, regex)
    ("white_spots", "White spots or patches on the shell", r"white\s+(spots?|patch(es)?|specks?|dots?|marks?|circles?)"),
    ("red_colour", "Reddish or pink body colour", r"\b(red|reddish|pink|pinkish)\b"),
    ("discolour", "Faded, opaque or abnormal body colour", r"\b(opaque|faded|pale|discolou?r\w*|darkened|whitish|milky|chalky|cloudy\s+body|dull\s+colou?r)\b"),
    ("soft_shell", "Soft or loose shell", r"soft(en\w*)?\s*(-\s*)?shells?|shells?\s+(are|is|become|becomes)?\s*(very\s+)?(soft|loose)|loose\s+(shell|cuticle)|shell[\s-]+meat\s+separation|softening"),
    ("lethargy", "Weak, sluggish or lethargic", r"\b(letharg\w*|sluggish|weak\w*|inactive|listless|lazy|dull)\b|reduced\s+(activity|swimming)|(not|less|hardly)\s+active|(very|too|moving)\s+slow(ly)?|slow\s+(and|moving|movement)"),
    ("feeding", "Reduced feeding or loss of appetite", r"(loss\s+of|reduced|poor|less|low|no)\s+(appetite|feeding|feed\s+intake)|(not|no\s+longer|hardly)\s+(eating|feeding)|stop\w*\s+(eating|feeding)|reluctan\w*\s+to\s+feed|empty\s+(gut|guts|stomach)|(gut|guts|stomach)\s+(is|are|look|looks)\s+empty|off\s+(feed|food)|feed\s+(is\s+)?(left|remaining|uneaten)|uneaten\s+feed"),
    ("swimming", "Abnormal swimming (erratic, at surface or pond edge)", r"\berratic\b|(swim\w*|float\w*|gather\w*|crowd\w*|congregat\w*|com\w*\s+up)\s+(near|at|on|along|to|by)\s+(the\s+)?(water\s+|pond\s+)?(surface|edge|edges|bank|banks|side|sides|dyke|corner|corners)|\bfloat(ing|s)?\b|without\s+orientation|in\s+circles|spiral\w*|spinning|tumbling"),
    ("mortality", "Mortality or sudden deaths", r"mortalit\w*|\bdying\b|\bdead\b|\bdie[sd]?\b|die[\s-]?off|\bdeaths?\b|mass\s+death|heavy\s+loss(es)?"),
    ("moulting", "Moulting problems", r"(can\s*not|cannot|can't|unable\s+to|fail\w*\s+to|abnormal|incomplete|unsuccessful|failed|problems?\s+(with|in))\s+(to\s+)?mou?lt\w*|mou?lt\w*\s+(is\s+|are\s+|has\s+)?(fail\w*|incomplete|abnormal|stuck|not\s+complet\w*)|stuck\s+in\s+(the\s+|their\s+)?(old\s+)?shells?"),
    ("necrosis", "Necrosis or dead tissue", r"necro\w*"),
    ("hepatopancreas", "Hepatopancreas abnormality", r"hepatopancreas"),
    ("posture", "Raised head or abnormal posture", r"rais\w*\s+(their\s+|its\s+)?heads?|\barching\b|\bhunching\b|abnormal\s+(body\s+)?posture"),
    ("growth", "Slow growth or weight loss", r"slow\s+growth|stunted|emaciat\w*|reduced\s+body\s+weight|weight\s+loss"),
]

NEGATION_RE = re.compile(r"\b(no|not|without|never|none|absence\s+of|free\s+of|nor|zero)\b[^.;,]{0,30}$", re.IGNORECASE)


@dataclass
class SymptomHit:
    key: str
    label: str
    matched_text: str
    start: int
    end: int
    negated: bool


def find_symptoms(text: str) -> list[SymptomHit]:
    text = normalise_text(text)
    hits: list[SymptomHit] = []
    for key, label, pattern in SYMPTOMS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            before = text[max(0, match.start() - 40) : match.start()]
            negated = bool(NEGATION_RE.search(before))
            hits.append(SymptomHit(key, label, match.group(0), match.start(), match.end(), negated))
            break
    return hits
