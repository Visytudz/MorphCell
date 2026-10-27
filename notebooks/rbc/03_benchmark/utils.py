import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def ensure_2d_features(features, feature_name="features"):
    """Validate feature arrays and coerce 1D inputs to shape (N, 1)."""
    features = np.asarray(features)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.ndim != 2:
        raise ValueError(
            f"{feature_name} must have shape (N, D) or (N,), got {features.shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError(f"{feature_name} contains NaN or inf values")
    return features


def ensure_labels(labels):
    """Validate label array shape and values."""
    labels = np.asarray(labels).squeeze()
    if labels.ndim != 1:
        raise ValueError(f"labels must have shape (N,), got {labels.shape}")
    if not np.isfinite(labels).all():
        raise ValueError("labels contain NaN or inf values")
    return labels


def validate_feature_label_pair(features, labels, feature_name="features"):
    """Validate feature/label pair and return normalized numpy arrays."""
    features = ensure_2d_features(features, feature_name=feature_name)
    labels = ensure_labels(labels)
    if len(features) != len(labels):
        raise ValueError(
            f"{feature_name} and labels must have the same number of samples: "
            f"{len(features)} vs {len(labels)}"
        )
    return features, labels


def align_feature_rows(features, sample_ids, reference_ids, feature_name="features"):
    """Align an array to the reference IDs, rejecting duplicate or missing samples."""
    features = ensure_2d_features(features, feature_name=feature_name)
    sample_ids = np.asarray(sample_ids).astype(str)
    reference_ids = np.asarray(reference_ids).astype(str)
    for name, ids in (("source", sample_ids), ("reference", reference_ids)):
        if ids.ndim != 1:
            raise ValueError(f"{feature_name}: {name} IDs must be one-dimensional")
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"{feature_name}: duplicate {name} IDs")
    if len(features) != len(sample_ids):
        raise ValueError(f"{feature_name}: feature rows and source IDs differ in length")
    if set(sample_ids) != set(reference_ids):
        missing = sorted(set(reference_ids) - set(sample_ids))
        extra = sorted(set(sample_ids) - set(reference_ids))
        raise ValueError(
            f"{feature_name}: sample IDs differ; missing={missing[:10]}, extra={extra[:10]}"
        )
    index = {sample_id: row for row, sample_id in enumerate(sample_ids)}
    return features[[index[sample_id] for sample_id in reference_ids]]


def load_feature_output(output_dir, filename, reference_ids, reference_labels):
    """Read an extraction output, align sample IDs, and check class labels."""
    output_dir = Path(output_dir)
    sample_ids = np.load(output_dir / "ids.npy")
    features = align_feature_rows(
        np.load(output_dir / filename), sample_ids, reference_ids,
        feature_name=output_dir.name,
    )
    labels = align_feature_rows(
        ensure_labels(np.load(output_dir / "labels.npy")),
        sample_ids, reference_ids, feature_name=f"{output_dir.name} labels",
    ).ravel()
    if not np.array_equal(labels, ensure_labels(reference_labels)):
        raise ValueError(f"{output_dir.name}: labels do not match the reference IDs")
    return features


def summarize_labels(labels, label2name=None):
    """Return a dataframe with class counts."""
    labels = ensure_labels(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    summary = pd.DataFrame({"label": unique_labels, "count": counts})
    if label2name is not None:
        summary["class_name"] = [
            label2name.get(label, str(label)) for label in unique_labels
        ]
    return summary.sort_values("label").reset_index(drop=True)


def build_analysis_features(
    features,
    scales=None,
    lambda_weight=0.0,
    apply_pca=False,
    pca_components=30,
    return_transformers=False,
):
    """
    Build the analysis feature space used for quantitative evaluation.
    PCA-compressed features are kept as-is without post-standardization.
    Scale is z-scored exactly once before fusion so that lambda retains meaning.
    """
    analysis_features = ensure_2d_features(features)
    fitted_pca = None
    fitted_scale_scaler = None

    if apply_pca:
        fitted_pca = PCA(n_components=pca_components, random_state=42)
        analysis_features = fitted_pca.fit_transform(analysis_features)

    if scales is not None:
        scales = ensure_2d_features(scales, feature_name="scales")
        if len(scales) != len(analysis_features):
            raise ValueError("scales and features must have the same number of samples")
        fitted_scale_scaler = StandardScaler()
        normalized_scales = fitted_scale_scaler.fit_transform(scales)
        if lambda_weight == 0:
            analysis_features = analysis_features.copy()
        else:
            analysis_features = np.concatenate(
                [analysis_features, lambda_weight * normalized_scales], axis=1
            )

    if return_transformers:
        return analysis_features, fitted_pca, fitted_scale_scaler
    return analysis_features


def build_knn_pipeline(n_neighbors=5, standardize_for_knn=False):
    """Build a kNN pipeline with optional standardization."""
    steps = []
    if standardize_for_knn:
        steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "knn",
            KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance"),
        )
    )
    return Pipeline(steps)


def evaluate_knn_cv(
    features,
    labels,
    n_neighbors=5,
    n_splits=5,
    n_repeats=10,
    random_state=42,
    standardize_for_knn=False,
):
    """Evaluate a feature set with repeated stratified kNN cross-validation."""
    features, labels = validate_feature_label_pair(features, labels)
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    base_pipeline = build_knn_pipeline(
        n_neighbors=n_neighbors,
        standardize_for_knn=standardize_for_knn,
    )

    records = []
    y_true_all = []
    y_pred_all = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(features, labels), start=1
    ):
        pipeline = clone(base_pipeline)
        pipeline.fit(features[train_idx], labels[train_idx])
        preds = pipeline.predict(features[test_idx])
        y_true = labels[test_idx]

        records.append(
            {
                "fold": fold_idx,
                "balanced_accuracy": balanced_accuracy_score(y_true, preds),
                "macro_f1": f1_score(y_true, preds, average="macro"),
            }
        )
        y_true_all.append(y_true)
        y_pred_all.append(preds)

    fold_df = pd.DataFrame(records)
    summary = {
        "balanced_accuracy_mean": fold_df["balanced_accuracy"].mean(),
        "balanced_accuracy_std": fold_df["balanced_accuracy"].std(ddof=1),
        "macro_f1_mean": fold_df["macro_f1"].mean(),
        "macro_f1_std": fold_df["macro_f1"].std(ddof=1),
        "n_folds": len(fold_df),
    }

    return {
        "fold_metrics": fold_df,
        "summary": summary,
        "y_true": np.concatenate(y_true_all),
        "y_pred": np.concatenate(y_pred_all),
    }


def evaluate_feature_sets(
    feature_registry,
    labels,
    n_neighbors=5,
    n_splits=5,
    n_repeats=10,
    random_state=42,
):
    """Run the same evaluation protocol for multiple feature sets."""
    labels = ensure_labels(labels)
    summary_rows = []
    detailed_results = {}

    for name, config in feature_registry.items():
        if isinstance(config, dict):
            features = config["features"]
            standardize_for_knn = config.get("standardize_for_knn", False)
        else:
            features = config
            standardize_for_knn = False

        features, local_labels = validate_feature_label_pair(
            features, labels, feature_name=name
        )
        eval_result = evaluate_knn_cv(
            features=features,
            labels=local_labels,
            n_neighbors=n_neighbors,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
            standardize_for_knn=standardize_for_knn,
        )
        summary_rows.append(
            {
                "feature_set": name,
                **eval_result["summary"],
                "feature_dim": features.shape[1],
                "standardize_for_knn": standardize_for_knn,
            }
        )
        detailed_results[name] = {
            **eval_result,
            "standardize_for_knn": standardize_for_knn,
        }

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="balanced_accuracy_mean", ascending=False
    )
    return summary_df, detailed_results


def plot_metric_comparison(
    summary_df,
    metrics=("balanced_accuracy_mean", "macro_f1_mean"),
    save_path=None,
    figsize=(12, 5),
    dpi=300,
):
    """Plot bar charts comparing the main evaluation metrics across feature sets."""
    summary_df = summary_df.copy()
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    error_map = {
        "balanced_accuracy_mean": "balanced_accuracy_std",
        "macro_f1_mean": "macro_f1_std",
    }

    for ax, metric in zip(axes, metrics):
        error_col = error_map.get(metric)
        plot_df = summary_df.sort_values(by=metric, ascending=False)
        ax.bar(
            plot_df["feature_set"],
            plot_df[metric],
            yerr=plot_df[error_col] if error_col in plot_df.columns else None,
            color="steelblue",
            alpha=0.85,
            capsize=4,
        )
        ax.set_title(metric.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✓ Metric comparison saved to: {save_path}")
    plt.show()


def plot_confusion_matrix(
    y_true,
    y_pred,
    label2name=None,
    normalize=True,
    title="Confusion Matrix",
    save_path=None,
    figsize=(8, 6),
    dpi=300,
):
    """Plot a confusion matrix from concatenated cross-validation predictions."""
    labels_sorted = np.unique(y_true)
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels_sorted,
        normalize="true" if normalize else None,
    )
    tick_labels = (
        [label2name.get(l, str(l)) for l in labels_sorted]
        if label2name
        else labels_sorted
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✓ Confusion matrix saved to: {save_path}")
    plt.show()
    return cm


def save_results_table(summary_df, save_path):
    """Save the summary metrics table to CSV."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(save_path, index=False)
    print(f"✓ Results table saved to: {save_path}")
