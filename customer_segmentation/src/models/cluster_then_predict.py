"""Two-stage baseline: cluster first, then predict promotion response."""
from typing import Tuple
import pandas as pd
from sklearn.linear_model import LogisticRegression


def fit_per_cluster_classifier(train_features: pd.DataFrame, train_labels: pd.Series) -> LogisticRegression:
    """Train a logistic regression classifier for a single cluster."""
    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(train_features, train_labels)
    return clf


def predict_with_clusters(
    cluster_labels: pd.Series,
    features: pd.DataFrame,
    classifiers: dict,
) -> Tuple[pd.Series, pd.Series]:
    """Predict responses per cluster using pre-trained classifiers."""
    preds = pd.Series(index=features.index, dtype=float)
    for cluster_id, clf in classifiers.items():
        idx = cluster_labels[cluster_labels == cluster_id].index
        if len(idx) == 0:
            continue
        preds.loc[idx] = clf.predict_proba(features.loc[idx])[:, 1]
    binary = (preds >= 0.5).astype(int)
    return preds, binary
