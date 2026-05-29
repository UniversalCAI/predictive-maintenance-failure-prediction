from __future__ import annotations

import json
import math
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


SEED = 42
FP_COST = 10
FN_COST = 500
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
MOD = OUT / "models"

DATA_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv",
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
]

FAILURE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
TARGET = "Machine failure"


@dataclass
class FitResult:
    name: str
    estimator: BaseEstimator
    val_score: np.ndarray
    test_score: np.ndarray
    fit_seconds: float


@dataclass
class OOFStackingModel:
    bases: list[tuple[str, BaseEstimator]]
    meta: BaseEstimator


def ensure_dirs() -> None:
    for path in [DATA_RAW, OUT, FIG, TAB, MOD]:
        path.mkdir(parents=True, exist_ok=True)


def download_data() -> Path:
    csv_path = DATA_RAW / "ai4i2020.csv"
    if csv_path.exists():
        return csv_path

    errors: list[str] = []
    for url in DATA_URLS:
        try:
            if url.endswith(".csv"):
                urllib.request.urlretrieve(url, csv_path)
                return csv_path
            zip_path = DATA_RAW / "ai4i2020.zip"
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
                if not members:
                    raise RuntimeError("zip file has no csv")
                with zf.open(members[0]) as src, csv_path.open("wb") as dst:
                    dst.write(src.read())
            return csv_path
        except Exception as exc:  # pragma: no cover
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Failed to download AI4I data:\n" + "\n".join(errors))


def load_data() -> pd.DataFrame:
    csv_path = download_data()
    df = pd.read_csv(csv_path)
    expected = {TARGET, *FAILURE_COLS}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [TARGET, *FAILURE_COLS]
    for col in ["UDI", "UID", "Product ID"]:
        if col in df.columns:
            drop_cols.append(col)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    y = df[TARGET].astype(int).copy()
    return X, y


def add_physics_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add mechanism-inspired features derived only from available sensors."""
    out = X.copy()
    eps = 1e-6
    air = out.get("Air temperature [K]")
    proc = out.get("Process temperature [K]")
    speed = out.get("Rotational speed [rpm]")
    torque = out.get("Torque [Nm]")
    wear = out.get("Tool wear [min]")

    if air is not None and proc is not None:
        out["temp_delta"] = proc - air
        out["temp_ratio"] = proc / (air + eps)
    if speed is not None and torque is not None:
        out["power_proxy"] = speed * torque
        out["torque_per_speed"] = torque / (speed + eps)
        out["speed_torque_ratio"] = speed / (torque + eps)
    if wear is not None and torque is not None:
        out["wear_torque"] = wear * torque
        out["wear_torque_ratio"] = wear / (torque + eps)
    if wear is not None and speed is not None:
        out["wear_speed"] = wear * speed
    if wear is not None and proc is not None:
        out["wear_process_temp"] = wear * proc
    if air is not None and proc is not None and speed is not None:
        out["thermal_speed_load"] = (proc - air) * speed
    return out


def get_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    num_cols, cat_cols = get_columns(X)
    num_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", num_step, num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(model: BaseEstimator, X: pd.DataFrame, scale_numeric: bool) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(X, scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


def safe_score(estimator: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        prob = estimator.predict_proba(X)
        return prob[:, 1]
    if hasattr(estimator, "decision_function"):
        raw = estimator.decision_function(X)
        raw = np.asarray(raw, dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    pred = estimator.predict(X)
    return np.asarray(pred, dtype=float)


def total_cost(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(FP_COST * fp + FN_COST * fn)


def metric_row(model: str, split: str, y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    row = {
        "model": model,
        "split": split,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "cost": total_cost(y_true, pred),
    }
    try:
        row["roc_auc"] = roc_auc_score(y_true, score)
    except ValueError:
        row["roc_auc"] = np.nan
    try:
        row["pr_auc"] = average_precision_score(y_true, score)
    except ValueError:
        row["pr_auc"] = np.nan
    return row


def optimize_threshold(y_true: np.ndarray, score: np.ndarray) -> dict:
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for th in thresholds:
        pred = (score >= th).astype(int)
        rows.append(
            {
                "threshold": float(th),
                "f1": f1_score(y_true, pred, zero_division=0),
                "mcc": matthews_corrcoef(y_true, pred),
                "recall": recall_score(y_true, pred, zero_division=0),
                "precision": precision_score(y_true, pred, zero_division=0),
                "cost": total_cost(y_true, pred),
            }
        )
    frame = pd.DataFrame(rows)
    best_cost = frame.sort_values(["cost", "mcc", "f1"], ascending=[True, False, False]).iloc[0]
    best_f1 = frame.sort_values(["f1", "mcc"], ascending=[False, False]).iloc[0]
    best_mcc = frame.sort_values(["mcc", "f1"], ascending=[False, False]).iloc[0]
    return {
        "grid": frame,
        "best_cost_threshold": float(best_cost.threshold),
        "best_f1_threshold": float(best_f1.threshold),
        "best_mcc_threshold": float(best_mcc.threshold),
    }


def xgb_model(weight: float | None = None, custom_obj: Callable | None = None) -> BaseEstimator:
    if XGBClassifier is None:
        raise RuntimeError("xgboost not available")
    params = dict(
        n_estimators=260,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        min_child_weight=2.0,
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    if weight is not None:
        params["scale_pos_weight"] = weight
    if custom_obj is not None:
        params["objective"] = custom_obj
    return XGBClassifier(**params)


def lgbm_model(weight: float | None = None) -> BaseEstimator:
    if LGBMClassifier is None:
        raise RuntimeError("lightgbm not available")
    params = dict(
        n_estimators=320,
        num_leaves=15,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        min_child_samples=15,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    if weight is not None:
        params["scale_pos_weight"] = weight
    return LGBMClassifier(**params)


def asymmetric_xgb_objective(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.clip(y_pred, -20, 20)
    prob = 1.0 / (1.0 + np.exp(-pred))
    # Positive failures are costlier. This is a stable custom objective for weighted BCE.
    weight = np.where(y_true == 1.0, 12.0, 1.0)
    grad = (prob - y_true) * weight
    hess = np.maximum(prob * (1.0 - prob) * weight, 1e-6)
    return grad, hess


def build_models(X: pd.DataFrame, y_train: pd.Series) -> dict[str, Pipeline]:
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    scale_pos = neg / max(pos, 1)
    models: dict[str, Pipeline] = {}

    models["Dummy_All_Normal"] = make_pipeline(DummyClassifier(strategy="most_frequent"), X, False)
    models["LogisticRegression"] = make_pipeline(LogisticRegression(max_iter=3000, random_state=SEED), X, True)
    models["LogisticRegression_Balanced"] = make_pipeline(
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED), X, True
    )
    models["KNN"] = make_pipeline(KNeighborsClassifier(n_neighbors=9, weights="distance"), X, True)
    models["LinearSVM_Balanced"] = make_pipeline(
        CalibratedClassifierCV(
            estimator=LinearSVC(class_weight="balanced", C=0.8, random_state=SEED, dual="auto"),
            cv=3,
            method="sigmoid",
        ),
        X,
        True,
    )
    models["DecisionTree_Balanced"] = make_pipeline(
        DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=SEED),
        X,
        False,
    )
    models["RandomForest_Balanced"] = make_pipeline(
        RandomForestClassifier(
            n_estimators=260,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        X,
        False,
    )
    if XGBClassifier is not None:
        models["XGBoost"] = make_pipeline(xgb_model(), X, False)
        models["XGBoost_Weighted"] = make_pipeline(xgb_model(scale_pos), X, False)
        models["XGBoost_AsymmetricLoss"] = make_pipeline(xgb_model(custom_obj=asymmetric_xgb_objective), X, False)
    if LGBMClassifier is not None:
        models["LightGBM"] = make_pipeline(lgbm_model(), X, False)
        models["LightGBM_Weighted"] = make_pipeline(lgbm_model(scale_pos), X, False)
    return models


def build_physics_models(X: pd.DataFrame, y_train: pd.Series) -> dict[str, Pipeline]:
    selected = {
        "RandomForest_Balanced",
        "XGBoost",
        "XGBoost_Weighted",
        "XGBoost_AsymmetricLoss",
        "LightGBM",
        "LightGBM_Weighted",
    }
    return {f"{name}_Physics": model for name, model in build_models(X, y_train).items() if name in selected}


def fit_and_score(
    name: str,
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> FitResult:
    start = time.time()
    fitted = clone(estimator)
    fitted.fit(X_train, y_train)
    fit_seconds = time.time() - start
    return FitResult(name, fitted, safe_score(fitted, X_val), safe_score(fitted, X_test), fit_seconds)


def make_anomaly_feature(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prep = make_preprocessor(X_train, scale_numeric=True)
    train_num = prep.fit_transform(X_train)
    val_num = prep.transform(X_val)
    test_num = prep.transform(X_test)
    normal_train = train_num[np.asarray(y_train) == 0]
    iso = IsolationForest(n_estimators=220, contamination="auto", random_state=SEED, n_jobs=-1)
    iso.fit(normal_train)

    def add_score(X: pd.DataFrame, arr: np.ndarray) -> pd.DataFrame:
        out = X.copy()
        out["anomaly_score"] = -iso.decision_function(arr)
        return out

    return add_score(X_train, train_num), add_score(X_val, val_num), add_score(X_test, test_num)


def fit_oof_stacking(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    model_factory: Callable[[pd.DataFrame, pd.Series], dict[str, Pipeline]],
    name: str,
) -> FitResult:
    base_names = ["LogisticRegression_Balanced", "LinearSVM_Balanced", "RandomForest_Balanced"]
    if XGBClassifier is not None:
        base_names.append("XGBoost_Weighted")
    if LGBMClassifier is not None:
        base_names.append("LightGBM_Weighted")

    all_models = model_factory(X_train, y_train)
    bases = [(n, all_models[n]) for n in base_names if n in all_models]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X_train), len(bases)))
    val_meta = np.zeros((len(X_val), len(bases)))
    test_meta = np.zeros((len(X_test), len(bases)))
    fitted_bases: list[tuple[str, BaseEstimator]] = []
    start = time.time()

    for j, (base_name, base_est) in enumerate(bases):
        fold_scores = np.zeros(len(X_train))
        for train_idx, hold_idx in cv.split(X_train, y_train):
            est = clone(base_est)
            est.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            fold_scores[hold_idx] = safe_score(est, X_train.iloc[hold_idx])
        oof[:, j] = fold_scores
        full_est = clone(base_est)
        full_est.fit(X_train, y_train)
        fitted_bases.append((base_name, full_est))
        val_meta[:, j] = safe_score(full_est, X_val)
        test_meta[:, j] = safe_score(full_est, X_test)

    meta = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)
    meta.fit(oof, y_train)
    fit_seconds = time.time() - start

    stacked = OOFStackingModel(fitted_bases, meta)
    val_score = meta.predict_proba(val_meta)[:, 1]
    test_score = meta.predict_proba(test_meta)[:, 1]
    return FitResult(name, stacked, val_score, test_score, fit_seconds)


def save_eda(df: pd.DataFrame) -> None:
    counts = df[TARGET].value_counts().rename_axis("class").reset_index(name="count")
    counts["label"] = counts["class"].map({0: "Normal", 1: "Failure"})
    counts.to_csv(TAB / "class_distribution.csv", index=False)

    failure_counts = df[FAILURE_COLS].sum().sort_values(ascending=False).reset_index()
    failure_counts.columns = ["failure_type", "count"]
    failure_counts.to_csv(TAB / "failure_type_distribution.csv", index=False)

    plt.figure(figsize=(5.5, 4))
    sns.barplot(data=counts, x="label", y="count", hue="label", legend=False)
    plt.title("Machine failure class distribution")
    plt.tight_layout()
    plt.savefig(FIG / "class_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.barplot(data=failure_counts, x="failure_type", y="count", hue="failure_type", legend=False)
    plt.title("Failure type distribution")
    plt.tight_layout()
    plt.savefig(FIG / "failure_type_distribution.png", dpi=180)
    plt.close()

    num_cols = [
        c
        for c in [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]
        if c in df.columns
    ]
    corr_cols = num_cols + [TARGET]
    plt.figure(figsize=(7, 5))
    sns.heatmap(df[corr_cols].corr(), cmap="RdBu_r", center=0, annot=True, fmt=".2f")
    plt.title("Numeric feature correlation")
    plt.tight_layout()
    plt.savefig(FIG / "feature_correlation.png", dpi=180)
    plt.close()


def plot_model_summary(results: pd.DataFrame) -> None:
    top = results.sort_values("mcc", ascending=False).head(10).copy()
    plt.figure(figsize=(9, 5))
    sns.barplot(data=top, y="model", x="mcc", hue="model", legend=False)
    plt.title("Top models by MCC")
    plt.tight_layout()
    plt.savefig(FIG / "top_models_mcc.png", dpi=180)
    plt.close()

    top_cost = results.sort_values("cost", ascending=True).head(10).copy()
    plt.figure(figsize=(9, 5))
    sns.barplot(data=top_cost, y="model", x="cost", hue="model", legend=False)
    plt.title("Top models by simulated maintenance cost")
    plt.tight_layout()
    plt.savefig(FIG / "top_models_cost.png", dpi=180)
    plt.close()


def plot_curves(
    y_test: pd.Series,
    scored: dict[str, np.ndarray],
    threshold_grid: pd.DataFrame,
    best_model: str,
) -> None:
    selected = list(scored.keys())[:6]
    plt.figure(figsize=(7, 5))
    for name in selected:
        fpr, tpr, _ = roc_curve(y_test, scored[name])
        auc = roc_auc_score(y_test, scored[name])
        plt.plot(fpr, tpr, label=f"{name} ({auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "roc_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    for name in selected:
        precision, recall, _ = precision_recall_curve(y_test, scored[name])
        ap = average_precision_score(y_test, scored[name])
        plt.plot(recall, precision, label=f"{name} ({ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "pr_curves.png", dpi=180)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(threshold_grid["threshold"], threshold_grid["cost"], color="tab:red", label="Cost")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Cost", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(threshold_grid["threshold"], threshold_grid["f1"], color="tab:blue", label="F1")
    ax2.plot(threshold_grid["threshold"], threshold_grid["mcc"], color="tab:green", label="MCC")
    ax2.set_ylabel("Score")
    plt.title(f"Threshold sweep: {best_model}")
    fig.tight_layout()
    plt.savefig(FIG / "threshold_sweep_best_model.png", dpi=180)
    plt.close()


def plot_confusion(y_test: pd.Series, score: np.ndarray, threshold: float, model_name: str) -> None:
    pred = (score >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Failure"], yticklabels=["Normal", "Failure"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(FIG / "best_model_confusion_matrix.png", dpi=180)
    plt.close()


def explain_tree_model(name: str, fitted: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    if not isinstance(fitted, Pipeline) or "model" not in fitted.named_steps:
        return
    model = fitted.named_steps["model"]
    preprocess = fitted.named_steps["preprocess"]
    X_trans = preprocess.transform(X_test)
    feature_names = preprocess.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        imp = imp.sort_values("importance", ascending=False).head(20)
        imp.to_csv(TAB / "tree_feature_importance.csv", index=False)
        plt.figure(figsize=(9, 6))
        sns.barplot(data=imp, y="feature", x="importance", hue="feature", legend=False)
        plt.title(f"Feature importance: {name}")
        plt.tight_layout()
        plt.savefig(FIG / "feature_importance.png", dpi=180)
        plt.close()

    if shap is not None and ("XGBoost" in name or "LightGBM" in name):
        try:
            sample_size = min(800, X_trans.shape[0])
            X_sample = X_trans[:sample_size]
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X_sample)
            if isinstance(values, list):
                values = values[-1]
            shap.summary_plot(values, X_sample, feature_names=feature_names, show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(FIG / "shap_summary.png", dpi=180, bbox_inches="tight")
            plt.close()
        except Exception as exc:
            (TAB / "shap_error.txt").write_text(str(exc), encoding="utf-8")
    else:
        try:
            perm = permutation_importance(fitted, X_test, y_test, n_repeats=5, random_state=SEED, n_jobs=-1, scoring="average_precision")
            perm_df = pd.DataFrame({"feature": X_test.columns, "importance": perm.importances_mean})
            perm_df.sort_values("importance", ascending=False).to_csv(TAB / "permutation_importance.csv", index=False)
        except Exception as exc:
            (TAB / "permutation_error.txt").write_text(str(exc), encoding="utf-8")


def summarize_failures(df: pd.DataFrame, y_test_idx: pd.Index, score: np.ndarray, threshold: float) -> None:
    test_df = df.loc[y_test_idx].copy()
    test_df["score"] = score
    test_df["pred"] = (score >= threshold).astype(int)
    test_df["error_type"] = np.select(
        [
            (test_df[TARGET] == 1) & (test_df["pred"] == 1),
            (test_df[TARGET] == 1) & (test_df["pred"] == 0),
            (test_df[TARGET] == 0) & (test_df["pred"] == 1),
        ],
        ["TP", "FN", "FP"],
        default="TN",
    )
    test_df.groupby("error_type")[FAILURE_COLS].sum().to_csv(TAB / "failure_type_by_error.csv")
    test_df.sort_values("score", ascending=False).head(20).to_csv(TAB / "top_risk_samples.csv", index=False)
    test_df[test_df["error_type"] == "FN"].to_csv(TAB / "false_negative_samples.csv", index=False)


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    df = load_data()
    save_eda(df)
    X, y = build_features(df)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=SEED, stratify=y_train_full
    )
    X_train_phys = add_physics_features(X_train)
    X_val_phys = add_physics_features(X_val)
    X_test_phys = add_physics_features(X_test)

    metadata = {
        "n_rows": int(len(df)),
        "n_features_used": int(X.shape[1]),
        "n_features_with_physics": int(X_train_phys.shape[1]),
        "physics_features": [c for c in X_train_phys.columns if c not in X.columns],
        "failure_rate": float(y.mean()),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "fp_cost": FP_COST,
        "fn_cost": FN_COST,
        "dropped_label_columns": FAILURE_COLS,
    }
    (TAB / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    model_defs = build_models(X_train, y_train)
    fit_results: list[FitResult] = []
    skipped: list[dict] = []
    for name, est in model_defs.items():
        try:
            print(f"[fit] {name}")
            fit_results.append(fit_and_score(name, est, X_train, y_train, X_val, X_test))
        except Exception as exc:
            skipped.append({"model": name, "reason": str(exc)})
            print(f"[skip] {name}: {exc}")

    physics_defs = build_physics_models(X_train_phys, y_train)
    for name, est in physics_defs.items():
        try:
            print(f"[fit] {name}")
            fit_results.append(fit_and_score(name, est, X_train_phys, y_train, X_val_phys, X_test_phys))
        except Exception as exc:
            skipped.append({"model": name, "reason": str(exc)})
            print(f"[skip] {name}: {exc}")

    print("[fit] OOF_Stacking")
    try:
        fit_results.append(fit_oof_stacking(X_train, y_train, X_val, X_test, build_models, "OOF_Stacking"))
    except Exception as exc:
        skipped.append({"model": "OOF_Stacking", "reason": str(exc)})
        print(f"[skip] OOF_Stacking: {exc}")

    print("[fit] OOF_Stacking_AnomalyFeature")
    try:
        X_train_a, X_val_a, X_test_a = make_anomaly_feature(X_train, y_train, X_val, X_test)
        fit_results.append(
            fit_oof_stacking(X_train_a, y_train, X_val_a, X_test_a, build_models, "OOF_Stacking_AnomalyFeature")
        )
    except Exception as exc:
        skipped.append({"model": "OOF_Stacking_AnomalyFeature", "reason": str(exc)})
        print(f"[skip] OOF_Stacking_AnomalyFeature: {exc}")

    print("[fit] OOF_Stacking_Physics")
    try:
        fit_results.append(
            fit_oof_stacking(X_train_phys, y_train, X_val_phys, X_test_phys, build_models, "OOF_Stacking_Physics")
        )
    except Exception as exc:
        skipped.append({"model": "OOF_Stacking_Physics", "reason": str(exc)})
        print(f"[skip] OOF_Stacking_Physics: {exc}")

    print("[fit] OOF_Stacking_Physics_AnomalyFeature")
    try:
        X_train_pa, X_val_pa, X_test_pa = make_anomaly_feature(X_train_phys, y_train, X_val_phys, X_test_phys)
        fit_results.append(
            fit_oof_stacking(
                X_train_pa,
                y_train,
                X_val_pa,
                X_test_pa,
                build_models,
                "OOF_Stacking_Physics_AnomalyFeature",
            )
        )
    except Exception as exc:
        skipped.append({"model": "OOF_Stacking_Physics_AnomalyFeature", "reason": str(exc)})
        print(f"[skip] OOF_Stacking_Physics_AnomalyFeature: {exc}")

    threshold_rows = []
    test_default_rows = []
    test_opt_rows = []
    test_f1_rows = []
    test_mcc_rows = []
    threshold_grids: dict[str, pd.DataFrame] = {}
    scores_for_curves: dict[str, np.ndarray] = {}
    fitted_by_name: dict[str, BaseEstimator] = {}

    for result in fit_results:
        opt = optimize_threshold(np.asarray(y_val), result.val_score)
        grid = opt["grid"].copy()
        grid.insert(0, "model", result.name)
        threshold_grids[result.name] = grid
        threshold_rows.append(
            {
                "model": result.name,
                "best_cost_threshold": opt["best_cost_threshold"],
                "best_f1_threshold": opt["best_f1_threshold"],
                "best_mcc_threshold": opt["best_mcc_threshold"],
                "fit_seconds": result.fit_seconds,
            }
        )
        test_default_rows.append(metric_row(result.name, "test_default_0.5", np.asarray(y_test), result.test_score, 0.5))
        test_opt_rows.append(
            metric_row(result.name, "test_cost_optimized", np.asarray(y_test), result.test_score, opt["best_cost_threshold"])
        )
        test_f1_rows.append(
            metric_row(result.name, "test_f1_optimized", np.asarray(y_test), result.test_score, opt["best_f1_threshold"])
        )
        test_mcc_rows.append(
            metric_row(result.name, "test_mcc_optimized", np.asarray(y_test), result.test_score, opt["best_mcc_threshold"])
        )
        scores_for_curves[result.name] = result.test_score
        fitted_by_name[result.name] = result.estimator

    thresholds = pd.DataFrame(threshold_rows)
    default_results = pd.DataFrame(test_default_rows).sort_values("mcc", ascending=False)
    opt_results = pd.DataFrame(test_opt_rows).sort_values(["cost", "mcc"], ascending=[True, False])
    f1_results = pd.DataFrame(test_f1_rows).sort_values(["f1", "mcc"], ascending=[False, False])
    mcc_results = pd.DataFrame(test_mcc_rows).sort_values(["mcc", "f1"], ascending=[False, False])
    grids = pd.concat(threshold_grids.values(), ignore_index=True) if threshold_grids else pd.DataFrame()

    thresholds.to_csv(TAB / "thresholds.csv", index=False)
    default_results.to_csv(TAB / "results_default_threshold.csv", index=False)
    opt_results.to_csv(TAB / "results_cost_optimized.csv", index=False)
    f1_results.to_csv(TAB / "results_f1_optimized.csv", index=False)
    mcc_results.to_csv(TAB / "results_mcc_optimized.csv", index=False)
    grids.to_csv(TAB / "threshold_sweep_all_models.csv", index=False)

    ablation_names = [
        ("A0_XGBoost_default", "XGBoost", default_results),
        ("A1_XGBoost_weighted_default", "XGBoost_Weighted", default_results),
        ("A2_XGBoost_weighted_cost_threshold", "XGBoost_Weighted", opt_results),
        ("A3_XGBoost_asymmetric_loss", "XGBoost_AsymmetricLoss", opt_results),
        ("A4_OOF_stacking_cost_threshold", "OOF_Stacking", opt_results),
        ("A5_OOF_stacking_anomaly_cost_threshold", "OOF_Stacking_AnomalyFeature", opt_results),
        ("A6_XGBoost_physics_cost_threshold", "XGBoost_Weighted_Physics", opt_results),
        ("A7_OOF_stacking_physics_cost_threshold", "OOF_Stacking_Physics", opt_results),
        ("A8_OOF_stacking_physics_anomaly_cost_threshold", "OOF_Stacking_Physics_AnomalyFeature", opt_results),
    ]
    ablation_rows = []
    for label, model_name, source in ablation_names:
        match = source[source["model"] == model_name]
        if not match.empty:
            row = match.iloc[0].to_dict()
            row["experiment"] = label
            ablation_rows.append(row)
    pd.DataFrame(ablation_rows).to_csv(TAB / "ablation_results.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(TAB / "skipped_models.csv", index=False)

    if opt_results.empty:
        raise RuntimeError("No successful model fits")

    best_model = opt_results.iloc[0]["model"]
    best_threshold = float(opt_results.iloc[0]["threshold"])
    best_score = scores_for_curves[best_model]

    best_summary = {
        "best_model_by_cost": best_model,
        "best_threshold": best_threshold,
        "test_metrics": opt_results.iloc[0].to_dict(),
    }
    (TAB / "best_model_summary.json").write_text(json.dumps(best_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_model_summary(opt_results)
    ordered_scores = {name: scores_for_curves[name] for name in opt_results["model"].head(6).tolist()}
    plot_curves(y_test, ordered_scores, threshold_grids[best_model], best_model)
    plot_confusion(y_test, best_score, best_threshold, best_model)

    # Explain the best single tree model, because stack explanations are less direct.
    explain_candidates = [m for m in opt_results["model"] if "XGBoost" in m or "LightGBM" in m]
    if explain_candidates:
        explain_name = explain_candidates[0]
        explain_X = X_test_phys if "Physics" in explain_name else X_test
        explain_tree_model(explain_name, fitted_by_name[explain_name], explain_X, y_test)
    else:
        explain_X = X_test_phys if "Physics" in best_model else X_test
        explain_tree_model(best_model, fitted_by_name[best_model], explain_X, y_test)

    summarize_failures(df, y_test.index, best_score, best_threshold)

    joblib.dump(fitted_by_name.get(best_model), MOD / "best_model.joblib")
    print(f"[done] best_model={best_model} threshold={best_threshold:.2f}")


if __name__ == "__main__":
    main()
