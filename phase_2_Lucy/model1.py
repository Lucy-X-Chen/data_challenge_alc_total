# XGBOOST
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
plt.style.use('fivethirtyeight')


def run_xgb(X_train, X_test, y_train, y_test, params, num_boost_round=10, tune_parameter=False, grid_search=False, graph=True):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # "Learn" the mean from the training data
    mean_train = np.mean(y_train)
    # Get predictions on the test set
    baseline_predictions = np.ones(y_test.shape) * mean_train
    # Compute MAE
    mae_baseline = mean_absolute_error(y_test, baseline_predictions)
    print("Baseline MAE is {:.2f}".format(mae_baseline))

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")],
        early_stopping_rounds=1000,
    )

    print("Best MAE: {:.5f} with {} rounds".format(
        model.best_score,
        model.best_iteration+1))

    if grid_search:
        params = build_grid_search(model, X_train, y_train, params)
        print('my new params are:', params)

    if tune_parameter:
        params = tune_parameters(model, dtrain, num_boost_round, params)
        print('my new params are:', params)

    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=30
    )

    cv_results['test-mae-mean'].min()

    num_boost_round = model.best_iteration + 10
    best_model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")])

    print("Best MAE: {:.5f} in {} rounds".format(
        model.best_score, model.best_iteration+1))
    print("MAE for best modeL: ", mean_absolute_error(
        best_model.predict(dtest), y_test))
    if graph:
        xgb.plot_importance(best_model)
        plt.rcParams['figure.figsize'] = [40, 40]
        plt.show()

    parameters = best_model.save_config()

    return best_model, parameters
   # for tuning parameters


def tune_parameters(model, dtrain, num_boost_round, params):
    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(11, 15)
        for min_child_weight in range(8, 11)
    ]

    min_mae = float("Inf")
    best_params = None
    for max_depth, min_child_weight in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}".format(
            max_depth,
            min_child_weight))
        # Update our parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        # Update best MAE
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (max_depth, min_child_weight)
    print("Best params: {}, {}, MAE: {}".format(
        best_params[0], best_params[1], min_mae))
    params['max_depth'] = best_params[0]
    params['min_child_weight'] = best_params[1]

    gridsearch_params = [
        (subsample, colsample)
        for subsample in [i/10. for i in range(4, 8)]
        for colsample in [i/10. for i in range(8, 12)]
    ]

    min_mae = float("Inf")
    best_params = None
    # We start by the largest values and go down to the smallest
    for subsample, colsample in reversed(gridsearch_params):
        print("CV with subsample={}, colsample={}".format(
            subsample,
            colsample))
        # We update our parameters
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        # Update best score
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (subsample, colsample)
    print("Best params: {}, {}, MAE: {}".format(
        best_params[0], best_params[1], min_mae))
    params['subsample'] = best_params[0]
    params['colsample'] = best_params[1]

    min_mae = float("Inf")
    best_params = None
    for eta in [.1, 0.05, 0.01]:  # alias: learning_rate
        print("CV with eta={}".format(eta))
        # We update our parameters
        params['eta'] = eta
        # Run and time CV
        cv_results = xgb.cv(params, dtrain, num_boost_round=num_boost_round,
                            seed=42, nfold=5, metrics=['mae'], early_stopping_rounds=10)
        # Update best score
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = eta
    print("Best params: {}, MAE: {}".format(best_params, min_mae))
    params['eta'] = best_params

    return params


def build_grid_search(model, X_train, y_train, params):
    parameters_for_testing = {
        'learning_rate': [0.08, 0.07, 0.09],
        'max_depth': [3, 5],
    }

    xgb_model = xgb.XGBRegressor()

    gsearch1 = GridSearchCV(estimator=xgb_model, param_grid=parameters_for_testing,
                            cv=5, n_jobs=4, verbose=2, scoring='neg_mean_absolute_error')
    gsearch1.fit(X_train, y_train)
    print(gsearch1.scorer_)
    print('best params')
    print(gsearch1.best_params_)
    print('best score')
    print(gsearch1.best_score_)
    return gsearch1.best_params_
