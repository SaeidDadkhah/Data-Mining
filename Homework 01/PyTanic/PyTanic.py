import pandas as pd
from pandas import DataFrame

import seaborn as sns
import matplotlib.pyplot as plt

# reading data
train = pd.read_csv("..\\input\\train.csv")
test = pd.read_csv("..\\input\\test.csv")

train.info()
print('------------------')
test.info()

train.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test.drop(['Name', 'Ticket'], axis=1)

# Embarked
train["Embarked"] = train["Embarked"].fillna("S")

g = sns.factorplot('Embarked', 'Survived', data=train, size=4, aspect=3)
sns.plt.show(g)

fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(15, 5))
sns.countplot(x='Embarked', data=train, ax=axis1)
sns.countplot(x='Survived', hue='Embarked', data=train, order=[1, 0], ax=axis2)

embark_survived = train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_survived, order=['S', 'C', 'Q'], ax=axis3)
sns.plt.show(fig)

embark_dummies_train = pd.get_dummies(train['Embarked'])
embark_dummies_train.drop(['S'], axis=1, inplace=True)
train = train.join(embark_dummies_train)
train.drop(['Embarked'], axis=1, inplace=True)

embark_dummies_test = pd.get_dummies(test['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)
test = test.join(embark_dummies_test)
test.drop(['Embarked'], axis=1, inplace=True)

# Fare
test['Fare'].fillna(test['Fare'].median(), inplace=True)

train['Fare'] = train['Fare'].astype(int)
test['Fare'] = test['Fare'].astype(int)

fare_not_survived = train['Fare'][train['Survived'] == 0]
fare_survived = train['Fare'][train['Survived'] == 1]

average_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])

fig = train['Fare'].plot(kind='hist', figsize=(15, 3), bins=100, xlim=(0, 50))
sns.plt.show(fig)

average_fare.index.names = std_fare.index.names = ['Survived']
fig = average_fare.plot(yerr=std_fare, kind='bar', legend=False)
sns.plt.show(fig)
