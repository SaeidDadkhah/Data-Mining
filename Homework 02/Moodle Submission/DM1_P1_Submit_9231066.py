import pandas as pd
import scipy.stats as dis
import numpy as np

import matplotlib.pyplot as plt

# from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_score

# from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.ensemble import RandomForestRegressor

from rmse import rmse
from impute import imputation

# Init
show_plots = False

generate_output = True

# Read data
raw_train = pd.read_csv('train.csv', header=None, names=range(1, 8))
test = pd.read_csv('test.csv', header=None, names=range(1, 7))
train = pd.DataFrame(raw_train.drop(7, axis=1))
y = raw_train[7]

print(train.head())


# Fill missing values
mean = []
std = []
t = []

fig, figs = plt.subplots(1, 6)
for i in range(1, 7):
    tmp = pd.DataFrame(train[i][train[i] != 0])
    tmp.hist(ax=figs[i-1], bins=10)
    mean.append(tmp.mean().get_value(i, 0))
    std.append(tmp.std())
    df = tmp.count()[i]
    t.append(dis.t(100, mean[-1], std[-1]))

if show_plots:
    plt.figure()
    plt.draw()

for index, row in train.iterrows():
    imputation(row, t)
for index, row in test.iterrows():
    imputation(row, t)

fig, figs = plt.subplots(1, 6)
for i in range(1, 7):
    train[i].hist(ax=figs[i - 1])
# plt.figure()
# plt.draw()

for i, j in zip(mean, train.mean()):
    print('{i}:{j}'.format(i=i, j=j))

# Dimension reduction
# selector = SelectKBest(f_classif, k=5)
# train = selector.fit_transform(X=train, y=y)
pca = PCA(n_components=5, whiten=True)
pca.fit(train, y)
train = pca.transform(train)
X_test = pca.transform(test)

# Train models
'''
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
# '''

print('Gradient Boosting:')
gbr = GradientBoostingRegressor()
score = (-cross_val_score(gbr, train, y, scoring='neg_mean_squared_error', cv=10)) ** 0.5
print(np.mean(score))

gbr.fit(train, y)
pred = gbr.predict(train)
print(rmse(pred, y))
'''
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

if generate_output:
    output = pd.DataFrame({
        1: test[1],
        2: test[2],
        3: test[3],
        4: test[4],
        5: test[5],
        6: test[6],
        7: gbr.predict(X_test)
    })
    output.to_csv('pca5gbr.csv', header=None, index=False)

if show_plots:
    plt.show()
