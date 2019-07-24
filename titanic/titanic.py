import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("train.csv")

df = df.drop(columns=['Name', 'Cabin', 'Embarked','Ticket'])

df = df.replace('male', 0)
df = df.replace('female', 1)
df = df.fillna(df.mean())

survived_passengers = df.loc[df['Survived'] == 1]

print(df)


training_set = df
training_set_label = training_set['Survived']
training_set = training_set.drop(columns=['Survived'])


df_test = pd.read_csv('test.csv')
df_test = df_test.drop(columns=['Name', 'Cabin', 'Embarked','Ticket'])
df_test = df_test.replace('male', 0)
df_test = df_test.replace('female', 1)
df_test = df_test.fillna(df.mean())
test_set = df_test

print(training_set)
print(training_set_label)
print(test_set)


forest_clf = DecisionTreeClassifier()
forest_clf.fit(training_set, training_set_label)

predicted = forest_clf.predict(test_set)
print(predicted)

df_test = df_test.drop(columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
df_test['Survived'] = predicted
print(df_test)
df_test.to_csv('solution.csv', index=False)
