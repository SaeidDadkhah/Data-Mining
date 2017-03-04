from sklearn.model_selection import KFold


def k_fold_cv(x, y, model, k):
    kf = KFold(n_splits=k)
    rmse = []
    for train, test in kf.split(x):
        x_train = x.loc[train]
        y_train = y.loc[train]
        x_test = x.loc[test]
        y_test = y.loc[test]

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rmse.append(rmse(y_test, predictions))
    return rmse

