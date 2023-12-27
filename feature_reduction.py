
from sklearn.feature_selection import RFECV
from ips_model_profiler.src.xgbclassifer import xgb_classifier


def recursive_feature_elimination(model, df, X, Y, min_features):
    rfecv = RFECV(model,
                  cv=5,
                  n_jobs=-1,
                  scoring='roc_auc',
                  min_features_to_select=min_features)

    rfecv = rfecv.fit(X, Y)

    # Best features
    cols = rfecv.get_support(indices=True)
    rfecv_features = X.columns[cols]

    print(f"The old model had {len(X.columns)} features")
    print(X.columns)

    X_new = df[rfecv_features]
    print(f"The new model has {len(X_new.columns)} features")
    print(X_new.columns)

    # Here we go
    new_model, accuracy, auc, f1score, _, _ = xgb_classifier(
        X_new,
        Y,
        n_estimators=600,
        plot_features=True)

    results = {"model": new_model,
               'accuracy': accuracy,
               'auc': auc,
               'f1_score': f1score}

    return results
