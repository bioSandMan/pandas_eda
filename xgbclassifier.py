
# If things don’t go your way in predictive modeling, use XGboost!
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import multiprocessing
import pandas as pd


def xgb_classifier(X,
                   Y,
                   useTrainCV=True,
                   cv_folds=5,
                   early_stopping_rounds=50,
                   learning_rate=0.1,
                   n_estimators=140,
                   max_depth=5,
                   min_child_weight=3,
                   gamma=0.2,
                   subsample=0.6,
                   colsample_bytree=1.0,
                   objective='binary:logistic',
                   scale_pos_weight=1,
                   useGridSearch=False,
                   param_test={},
                   show_plot=True,
                   verbose=True,
                   number_of_features_shown=10):

    ncpu = multiprocessing.cpu_count()
    # split data into train and test sets
    seed = 7
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        Y,
                                                        test_size=test_size,
                                                        random_state=seed)

    if useGridSearch:
        print("Performing Grid Search")
        if param_test:
            estimator = xgb.XGBClassifier(learning_rate=learning_rate,
                                          n_estimators=n_estimators,
                                          max_depth=max_depth,
                                          min_child_weight=min_child_weight,
                                          gamma=gamma,
                                          subsample=subsample,
                                          colsample_bytree=colsample_bytree,
                                          objective=objective,
                                          nthread=ncpu,
                                          scale_pos_weight=scale_pos_weight,
                                          seed=seed)
            alg = GridSearchCV(estimator=estimator,
                               aram_grid=param_test,
                               scoring='f1',
                               n_jobs=4, iid=False, cv=5)
            alg.fit(X_train, y_train)
            print(alg.cv_results_, alg.best_params_, alg.best_score_)
            return
        else:
            print("No parameters in param_test."
                  "Please provide parameters to test.")
    else:
        alg = xgb.XGBClassifier(learning_rate=learning_rate,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_child_weight=min_child_weight,
                                gamma=gamma,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                objective=objective,
                                nthread=ncpu,
                                scale_pos_weight=scale_pos_weight,
                                seed=seed)

    if useTrainCV:
        if verbose:
            print("Start Feeding Data")
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train.values, label=y_train.values)
        cvresult = xgb.cv(xgb_param,
                          xgtrain,
                          num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cv_folds,
                          metrics='auc',
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    if verbose:
        print('Start Training')
        alg.fit(X_train, y_train, eval_metric='auc')

        print("Start Predicting")
        predictions = alg.predict(X_test)
        pred_proba = alg.predict_proba(X_test)[:, 1]

        print("\n Finished")
        print("Accuracy : %.4g" % metrics.accuracy_score(y_test, predictions))
        print("AUC: %f" % metrics.roc_auc_score(y_test, pred_proba))
        print("F1 Score: %f" % metrics.f1_score(y_test, predictions))
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()
        informedness = (tp/(tp+fn)) - (fn/(fp+tn))
        print("Informedness {:0.2f}".format(informedness))

    else:
        alg.fit(X_train, y_train, eval_metric='auc')
        predictions = alg.predict(X_test)
        pred_proba = alg.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()
        informedness = (tp/(tp+fn)) - (fn/(fp+tn))

    if show_plot:
        feat_imp = alg.feature_importances_
        feat = X_train.columns.tolist()
        res_df = pd.DataFrame(
                              {'Features': feat, 'Importance': feat_imp}
                              ).sort_values(
                                  by='Importance', ascending=False
                                  ).head(number_of_features_shown)
        res_df.plot('Features',
                    'Importance',
                    kind='bar',
                    title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()
        print(res_df)

        # calculate the fpr and tpr for all thresholds of the classification
        fpr, tpr, threshold = metrics.roc_curve(y_test, pred_proba)
        roc_auc = metrics.auc(fpr, tpr)

        # method I: plt
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    else:
        feat_imp = alg.feature_importances_
        feat = X_train.columns.tolist()
        res_df = pd.DataFrame(
                              {'Features': feat, 'Importance': feat_imp}
                              ).sort_values(
                                  by='Importance', ascending=False
                                  ).head(number_of_features_shown)

    return alg, \
        metrics.accuracy_score(y_test, predictions), \
        metrics.roc_auc_score(y_test, pred_proba), \
        metrics.f1_score(y_test, predictions), informedness, res_df
