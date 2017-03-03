import pandas as pd
import scipy.stats as dis

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

raw_train = pd.read_csv('train.csv', header=None, names=range(1, 8))
train = pd.DataFrame(raw_train.drop(7, axis=1))

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
# sns.plt.show(fig)

for index, row in train.iterrows():
    zeros = []
    tmp = []
    for i in range(1, 7):
        if row.get_value(i, 0) == 0:
            zeros.append(i)
        else:
            tmp.append(t[i-1].cdf(row.get_value(i, 0)))
    if len(tmp) != 0:
        tmp = sum(tmp)/len(tmp)
        # print(tmp)
        # print(x)
        for i in zeros:
            res = t[i-1].ppf(tmp)
            res = min(res, 20)
            res = max(res, 0)
            row.set_value(i, value=res)
    else:
        for i in zeros:
            res = row.set_value(i, value=t[i-1].ppf(0.5))
        # print(x)

# '''
fig, figs = plt.subplots(1, 6)
for i in range(1, 7):
    train[i].hist(ax=figs[i - 1])
# sns.plt.show(fig)
# '''

for i, j in zip(mean, train.mean()):
    print('{i}:{j}'.format(i=i, j=j))

# Train models

y = raw_train[7]
lr = LinearRegression()
lr.fit(train, y)
pred = lr.predict(train)
print('Linear Regression: {x}'.format(x=sum(abs(pred-y) ** 2)))

for i in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
    lar = Lasso(alpha=i)
    lar.fit(train, y)
    pred = lar.predict(train)
    print('Lasso Regression (alpha: {i}): {x}'.format(i=i, x=sum(abs(pred-y) ** 2)))
    # print('Lasso Regression (alpha: {i}): {x}'.format(i=i, x=lar.score(train, y)))
    # if i == 0.03:
    #     pred = lar.predict(train)

gbr = GradientBoostingRegressor()
gbr.fit(train, y)
pred = gbr.predict(train)
print('Gradient Boosting: {x}'.format(x=sum(abs(pred-y) ** 2)))
# print('Gradient Boosting: {x}'.format(x=lar.score(train, y)))

abr = AdaBoostRegressor()
abr.fit(train, y)
pred = abr.predict(train)
print('AdaBoost: {x}'.format(x=sum(abs(pred-y) ** 2)))
# print('AdaBoost: {x}'.format(x=lar.score(train, y)))

rf = RandomForestRegressor()
rf.fit(train, y)
pred = rf.predict(train)
print('Random Forest: {x}'.format(x=sum(abs(pred-y) ** 2)))
# print('Random Forest: {x}'.format(x=lar.score(train, y)))

output = pd.DataFrame({
    1: train[1],
    2: train[2],
    3: train[3],
    4: train[4],
    5: train[5],
    6: train[6],
    7: rf.predict(train)
})
output.to_csv('output.csv')

print(sum(abs(pred-y) ** 2))
