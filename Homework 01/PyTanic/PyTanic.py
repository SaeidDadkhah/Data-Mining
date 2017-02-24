import pandas as pd
from pandas import DataFrame

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB


# Control behavior
sns.set_style('whitegrid')
stop_at_plots = False

# reading data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train.info()
print('------------------')
test.info()

train = train.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test = test.drop(['Name', 'Ticket'], axis=1)

# Embarked
train["Embarked"] = train["Embarked"].fillna("S")

fig = sns.factorplot('Embarked', 'Survived', data=train, size=4, aspect=3)
if stop_at_plots:
    sns.plt.show(fig)

fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(15, 5))
sns.countplot(x='Embarked', data=train, ax=axis1)
sns.countplot(x='Survived', hue='Embarked', data=train, order=[1, 0], ax=axis2)

embark_survived = train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_survived, order=['S', 'C', 'Q'], ax=axis3)
if stop_at_plots:
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
if stop_at_plots:
    sns.plt.show(fig)

average_fare.index.names = std_fare.index.names = ['Survived']
fig = average_fare.plot(yerr=std_fare, kind='bar', legend=False)
if stop_at_plots:
    sns.plt.show(fig)

# Age
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
axis1.set_title('Original Age Values')
axis2.set_title('New Age Values')

train_age_mean = train['Age'].mean()
train_age_std = train['Age'].std()
train_age_na = train['Age'].isnull().sum()
train_age_rand = np.random.randint(train_age_mean - train_age_std,
                                   train_age_mean + train_age_std,
                                   size=train_age_na)

test_age_mean = test['Age'].mean()
test_age_std = test['Age'].std()
test_age_na = test['Age'].isnull().sum()
test_age_rand = np.random.randint(test_age_mean - test_age_std,
                                  test_age_mean + test_age_std,
                                  size=test_age_na)

train['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

train['Age'][np.isnan(train['Age'])] = train_age_rand
train['Age'] = train['Age'].astype(int)

test['Age'][np.isnan(test['Age'])] = test_age_rand
test['Age'] = test['Age'].astype(int)

train['Age'].hist(bins=70, ax=axis2)
if stop_at_plots:
    sns.plt.show(fig)

facet = sns.FacetGrid(train, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
if stop_at_plots:
    sns.plt.show(facet)

fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))
train_age_mean = train[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()
fig = sns.barplot(x='Age', y='Survived', data=train_age_mean)
if stop_at_plots:
    sns.plt.show(fig)

# Cabin
train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)

# Family
train['Family'] = train['Parch'] + train['SibSp']
train['Family'].loc[train['Family'] > 0] = 1
train['Family'].loc[train['Family'] == 0] = 0

test['Family'] = test['Parch'] + test['SibSp']
test['Family'].loc[test['Family'] > 0] = 1
test['Family'].loc[test['Family'] == 0] = 0

fig, (axis1, axis2) = plt.subplots(1, 2, sharex=True, figsize=(10, 5))

sns.countplot(x='Family', data=train, order=[1, 0], ax=axis1)

family_prec = train[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=family_prec, order=[1, 0], ax=axis2)

axis1.set_xticklabels(['With Family', 'Alone'], rotation=0)
if stop_at_plots:
    sns.plt.show(fig)


# Sex
def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex


train['Person'] = train[['Age', 'Sex']].apply(get_person, axis=1)
train.drop(['Sex'], axis=1, inplace=True)

test['Person'] = test[['Age', 'Sex']].apply(get_person, axis=1)
test.drop(['Sex'], axis=1, inplace=True)

person_dummies_train = pd.get_dummies(train['Person'])
person_dummies_train.columns = ['Child', 'Female', 'Male']
person_dummies_train.drop(['Male'], axis=1, inplace=True)
train = train.join(person_dummies_train)

person_dummies_test = pd.get_dummies(test['Person'])
person_dummies_test.columns = ['Child', 'Female', 'Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)
test = test.join(person_dummies_test)

fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))

sns.countplot(x='Person', data=train, ax=axis1)

person_prec = train[['Person', 'Survived']].groupby(['Person'], as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=person_prec, ax=axis2, order=['male', 'female', 'child'])
sns.plt.show(fig)

train.drop(['Person'], axis=1, inplace=True)
test.drop(['Person'], axis=1, inplace=True)

# Pclass
fig = sns.factorplot('Pclass', 'Survived', order=[1, 2, 3], data=train, size=5)
if stop_at_plots:
    sns.plt.show(fig)

pclass_dummies_train = pd.get_dummies(train['Pclass'])
pclass_dummies_train.columns = ['Class_1', 'Class_2', 'Class_3']
pclass_dummies_train.drop(['Class_3'], axis=1, inplace=True)
train.drop(['Pclass'], axis=1, inplace=True)
train = train.join(pclass_dummies_train)

pclass_dummies_test = pd.get_dummies(test['Pclass'])
pclass_dummies_test.columns = ['Class_1', 'Class_2', 'Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)
test.drop(['Pclass'], axis=1, inplace=True)
test = test.join(pclass_dummies_test)

# define training and test sets
X_train = train.drop('Survived', axis=1)
Y_train = train['Survived']
X_test = test.drop('PassengerId', axis=1).copy()

# Logistic regression
'''
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_logreg = logreg.predict(X_test)
print('Logistic regression: {x}'.format(x=logreg.score(X_train, Y_train)))

# SVM
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svc = svc.predict(X_test)
print('SVC: {x}'.format(x=svc.score(X_train, Y_train)))

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred_lsvc = linear_svc.predict(X_test)
print('Linear SVC: {x}'.format(x=linear_svc.score(X_train, Y_train)))
'''

# Random Forests
random_forest = RandomForestClassifier()
random_forest.fit(X_train, Y_train)
Y_pred_rf = random_forest.predict(X_test)
print('Random Forest: {x}'.format(x=random_forest.score(X_train, Y_train)))

# KNN
'''
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
print('knn: {x}'.format(x=knn.score(X_train, Y_train)))

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred_gaussian = gaussian.predict(X_test)
print('Gaussian Naive Bayes: {x}'.format(x=gaussian.score(X_train, Y_train)))
'''

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": Y_pred_rf
})
submission.to_csv('PyTanic_rf0.csv', index=False)
