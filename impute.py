
# To use the experimental IterativeImputer, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import numpy as np


N_SPLITS = 5


def get_full_score(X, y, regressor):
    full_scores = cross_val_score(regressor, X, y,
                                  scoring='roc_auc',
                                  cv=N_SPLITS)
    return full_scores.mean(), full_scores.std()


def get_scores_for_imputer(imputer, X, y, regressor):
    X_zero = X.replace([np.inf, -np.inf], 0)
    estimator = make_pipeline(imputer, regressor)
    impute_scores = cross_val_score(estimator, X_zero, y,
                                    scoring='roc_auc',
                                    cv=N_SPLITS)
    return impute_scores


def replace_inf_nan(X, y, regressor):
    X_r = X.replace([np.inf, -np.inf], np.nan)
    scores = cross_val_score(regressor, X_r, y,
                             scoring='roc_auc',
                             cv=N_SPLITS)
    return scores.mean(), scores.std()


def replace_inf_zero(X, y, regressor):
    X_zero = X.replace([np.inf, -np.inf], 0)
    full_scores = cross_val_score(regressor, X_zero, y,
                                  scoring='roc_auc',
                                  cv=N_SPLITS)
    return full_scores.mean(), full_scores.std()


def get_impute_zero_score(X, y):
    imputer = SimpleImputer(missing_values=np.nan, add_indicator=True,
                            strategy='constant', fill_value=0)
    zero_impute_scores = get_scores_for_imputer(imputer, X, y)
    return zero_impute_scores.mean(), zero_impute_scores.std()


def get_impute_knn_score(X, y):
    imputer = KNNImputer(missing_values=np.nan, add_indicator=True)
    knn_impute_scores = get_scores_for_imputer(imputer, X, y)
    return knn_impute_scores.mean(), knn_impute_scores.std()


def get_impute_mean(X, y):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean",
                            add_indicator=True)
    mean_impute_scores = get_scores_for_imputer(imputer, X, y)
    return mean_impute_scores.mean(), mean_impute_scores.std()


def get_impute_iterative(X, y):
    imputer = IterativeImputer(missing_values=np.nan, add_indicator=True,
                               random_state=0, n_nearest_features=5,
                               sample_posterior=True)
    iterative_impute_scores = get_scores_for_imputer(imputer, X, y)
    return iterative_impute_scores.mean(), iterative_impute_scores.std()
