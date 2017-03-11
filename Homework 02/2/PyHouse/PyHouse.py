import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import variance_threshold, SelectKBest, f_regression
from sklearn.decomposition import PCA
# '''
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
# '''
from sklearn.ensemble import RandomForestRegressor

from rmse import rmse

# 0. Init script
show_figures = False

select_features = False
var_th = False
select_k_best = True
use_pca = True

generate_output = False

# 1. Load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# 2. Look at data correlation
if show_figures:
    threshold = 0.5

    correlation = train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'GarageYrBlt',
                         'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']].corr()
    sns.set(font_scale=1.10)
    sns.heatmap(correlation, vmax=0.8, linewidths=0.01, square=True, annot=True, cmap='viridis', linecolor='white')
    plt.xticks(rotation='vertical')
    plt.yticks(rotation=0)
    plt.title('Correlation between features')
    plt.figure()
    plt.draw()

# 3. Visualize sale price
if show_figures:
    sns.distplot(train['SalePrice'])
    plt.title('Distribution of Sale Price')
    plt.ylabel('Number of Occurences')
    plt.xlabel('Sale Price')
    plt.xticks(rotation=0)
    plt.figure()
    plt.draw()

# 4. Control outliers
upper_limit = np.percentile(train.SalePrice.values, 99.5)
train['SalePrice'].ix[train['SalePrice'] > upper_limit] = upper_limit

if show_figures:
    sns.distplot(train['SalePrice'])
    plt.title('Distribution of Sale Price')
    plt.ylabel('Number of Occurences')
    plt.xlabel('Sale Price')
    plt.xticks(rotation=0)
    plt.figure()
    plt.draw()

# 5. Missing value imputation
null_columns = train.columns[train.isnull().any()]

if show_figures:
    labels = []
    values = []
    for column in null_columns:
        labels.append(column)
        values.append(train[column].isnull().sum())
    ind = np.arange(len(labels))
    width = 0.9
    plt.barh(ind, np.array(values))
    plt.yticks(ind + (width / 2.), labels, rotation='horizontal')
    plt.xlabel('Count of  missing values')
    plt.ylabel('Column Names')
    plt.title('Variables with missing values')
    # plt.figure()
    # plt.draw()

# 5. 1. LotFrontage
print('LotFrontage ~ LotArea: {}'.format(train['LotFrontage'].corr(train['LotArea'])))
train['SqrtLotArea'] = np.sqrt(train['LotArea'])
print('LotFrontage ~ SqrtLotArea: {}'.format(train['LotFrontage'].corr(train['SqrtLotArea'])))

lot_filter = train['LotFrontage'].isnull()
train.LotFrontage[lot_filter] = train.SqrtLotArea[lot_filter]

test['SqrtLotArea'] = np.sqrt(test['LotArea'])
lot_filter = test['LotFrontage'].isnull()
test.LotFrontage[lot_filter] = test.SqrtLotArea[lot_filter]

# 5. 2. MasVnrType and MasVnrArea
train['MasVnrType'] = train['MasVnrType'].fillna('None')
test['MasVnrType'] = test['MasVnrType'].fillna('None')
train['MasVnrArea'] = train['MasVnrArea'].fillna(0.0)
test['MasVnrArea'] = test['MasVnrArea'].fillna(0.0)

# 5. 3. Electrical
train['Electrical'] = train['Electrical'].fillna('SBrkr')
test['Electrical'] = test['Electrical'].fillna('SBrkr')

# 5. 4. Alley
train['Alley'] = train['Alley'].fillna('None')
test['Alley'] = test['Alley'].fillna('None')

# 5. 5. Basement Features
upper_limit = np.percentile(train.TotalBsmtSF.values, 99.5)
train['TotalBsmtSF'].ix[train['TotalBsmtSF'] > upper_limit] = upper_limit
test['TotalBsmtSF'].ix[test['TotalBsmtSF'] > upper_limit] = upper_limit

basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2']
for column in basement_cols:
    if 'FinSF' not in column:
        train[column] = train[column].fillna('None')
        test[column] = test[column].fillna('None')

# 5. 6. Fireplaces
train['FireplaceQu'] = train['FireplaceQu'].fillna('None')
test['FireplaceQu'] = test['FireplaceQu'].fillna('None')

# 5. 7. Garages
upper_limit = np.percentile(train.GarageArea.values, 99.5)
train['GarageArea'].ix[train['GarageArea'] > upper_limit] = upper_limit
test['GarageArea'].ix[test['GarageArea'] > upper_limit] = upper_limit

garage_cols = ['GarageType', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea']
for column in garage_cols:
    if train[column].dtype == np.object:
        train[column] = train[column].fillna('None')
        test[column] = test[column].fillna('None')
    else:
        train[column] = train[column].fillna(0)
        test[column] = test[column].fillna(0)

# 5. 8. Pool
train['PoolQC'] = train['PoolQC'].fillna('None')
test['PoolQC'] = test['PoolQC'].fillna('None')

# 5. 9. Fence
train['Fence'] = train['Fence'].fillna('None')
test['Fence'] = test['Fence'].fillna('None')

# 5. 10. MiscFeature
train['MiscFeature'] = train['MiscFeature'].fillna('None')
test['MiscFeature'] = test['MiscFeature'].fillna('None')

# 5. 11. Impute other features
other_features = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                  'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual', 'Functional', 'SaleType']
for column in other_features:
    test[column] = test[column].fillna(train[column].mode()[0])

# 5. 12. Make features dummy
for column in train.columns:
    if train[column].dtype == np.object:
        dummies = pd.get_dummies(train[column])
        dummies.drop(dummies.columns[0], axis=1, inplace=True)
        columns = list(dummies.columns)
        train_columns = columns
        for i in range(0, len(columns)):
            columns[i] = column + columns[i]
        dummies.columns = columns
        train.drop(column, axis=1, inplace=True)
        train = train.join(dummies)

        dummies = pd.get_dummies(test[column])
        dummies.drop(dummies.columns[0], axis=1, inplace=True)
        columns = list(dummies.columns)
        for i in range(0, len(columns)):
            columns[i] = column + columns[i]
        dummies.columns = columns

        if len(columns) != len(train_columns):
            for col in train_columns:
                if col not in columns:
                    dummies[col] = 0

        test.drop(column, axis=1, inplace=True)
        test = test.join(dummies)

# print(train[null_columns].isnull().sum())
# null_columns = test.columns[test.isnull().any()]
# print(sum(test.isnull().sum()))

# 6. Select features
y = np.log(train['SalePrice'])
train.drop(['Id', 'SalePrice'], axis=1, inplace=True)
X_test = test.drop('Id', axis=1)

print(train.shape)

if select_features:
    if var_th:
        print('Selecting features: variance threshold')
        selector = variance_threshold.VarianceThreshold()
        train = selector.fit_transform(X=train, y=y)
        X_test = selector.transform(X_test)
    elif select_k_best:
        print('Selecting features: select k best')
        selector = SelectKBest(f_regression, k=175)
        train = selector.fit_transform(X=train, y=y)
        X_test = selector.transform(X=X_test)

# 7. PCA

# '''
if use_pca:
    data = pd.concat([train, X_test], ignore_index=True)
    pca = PCA(n_components=200, whiten=True)
    pca.fit(data)
    data = pca.transform(data)
    train = data[:1460]
    print(train.shape)
    X_test = data[1460:]
    print(X_test.shape)
# '''
print(train.shape)

# 8. Model Selection
# '''
print('Linear Regression:')
lr = LinearRegression()
score = (-cross_val_score(lr, train, y, scoring='neg_mean_squared_error', cv=10)) ** 0.5
print(np.mean(score))

lr.fit(train, y)
pred = lr.predict(train)
print(rmse(pred, y))

for i in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
    print('Lasso Regression (alpha: {i}):'.format(i=i))
    lar = Lasso(alpha=i)
    score = (-cross_val_score(lar, train, y, scoring='neg_mean_squared_error', cv=10)) ** 0.5
    print(np.mean(score))

    lar.fit(train, y)
    pred = lar.predict(train)
    print(rmse(pred, y))

print('Gradient Boosting:')
gbr = GradientBoostingRegressor()
score = (-cross_val_score(gbr, train, y, scoring='neg_mean_squared_error', cv=10)) ** 0.5
print(np.mean(score))

gbr.fit(train, y)
pred = gbr.predict(train)
print(rmse(pred, y))

print('AdaBoost:')
abr = AdaBoostRegressor()
score = (-cross_val_score(abr, train, y, scoring='neg_mean_squared_error', cv=10)) ** 0.5
print(np.mean(score))

abr.fit(train, y)
pred = abr.predict(train)
print(rmse(pred, y))

print('Random Forest:')
rf = RandomForestRegressor()
score = (-cross_val_score(rf, train, y, scoring='neg_mean_squared_error', cv=10)) ** 0.5
print(np.mean(score))

rf.fit(train, y)
pred = rf.predict(train)
print(rmse(pred, y))
# '''

# 9. Generate Output
if generate_output:
    y = np.exp(lar.predict(X_test))
    submission = pd.DataFrame({
        'Id': test['Id'],
        'SalePrice': y
    })
    submission.to_csv('lar0_001.csv', index=False)

# 10. Finalize
if show_figures:
    plt.show()
