"""
Cross-subject classification of the extracted PAMAP2 features.

Each estimator is scored by cross-validation that holds out one subject per fold
(LeaveOneGroupOut grouped by subject_id), so the scores reflect how the model generalizes
to a subject absent from training. The StandardScaler is a step in the pipeline, so
cross_validate fits it on the training fold alone and the held-out subject does not leak
into the scaling.
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_validate

# Columns of the feature matrix that are labels/metadata, not features.
LABEL_COLUMNS = ['subject_id', 'interval_id', 'activity_id']


def build_pipeline(estimator):
    """
    Chain a StandardScaler transformer and the estimator into one pipeline. StandardScaler
    recenters each feature to mean 0 and scales it to variance 1. As a pipeline step it is
    fit on the training fold alone, so the scaling carries no leakage. Tree estimators do
    not need it, but it is harmless there.
    """
    return Pipeline([('scale', StandardScaler()), ('model', estimator)])


def loso_scores(features_df, estimator, scoring=('accuracy', 'f1_weighted')):
    """
    Cross-validation scores for one estimator on the feature matrix from
    extract_all_features. Each fold holds out one subject (grouped by subject_id), so the
    scores measure generalization to a subject absent from training. activity_id is the
    target.
    """
    features = features_df.drop(columns=LABEL_COLUMNS)
    target = features_df['activity_id']
    subjects = features_df['subject_id']
    return cross_validate(build_pipeline(estimator), features, target,
                          groups=subjects, cv=LeaveOneGroupOut(), scoring=list(scoring))
