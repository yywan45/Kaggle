# CLEAN DATA -------------------------------------------------------------------------

# library to read data
import pandas

# use pandas to read csv file into data frame
titanic = pandas.read_csv("Data/train.csv")

# set null ages to median
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# set male and female to 0 and 1
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# # prints the unique categories in Embarked
# print(titanic["Embarked"].unique())

# clean Embarked data
titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic.loc[titanic["Embarked"] == 'S', "Embarked"] = 0
titanic.loc[titanic["Embarked"] == 'C', "Embarked"] = 1
titanic.loc[titanic["Embarked"] == 'Q', "Embarked"] = 2

# print top 10 rows
print(titanic.head(30))

# # print description of all data
# print(titanic.describe())


# RANDOM FOREST 1 -------------------------------------------------------------------------

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

# make 3 folds
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

# get scores, note that in cv, we use the cv keyword
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
print(scores.mean())

# RANDOM FOREST 2 -------------------------------------------------------------------------

# train more trees - takes more time, but will increase accuracy up to a point
# increase split and leaf to reduce overfitting
alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)

# make 3 folds
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

# get scores, note that in cv, we use the cv keyword
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

# GET TITLES -------------------------------------------------------------------------

import re

# function to get the title from a name
def get_title(name):
    # use regex to search for a title
    # titles always consist of capital and lowercase letters, and end with a period
    title_search = re.search(' ([A-Za-z]+)\.', name)

    # if title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# get all titles and print how often each one occurs
titles = titanic["Name"].apply(get_title)
print(pandas.value_counts(titles))

# map each title to an integer
# rare titles are compressed into the same codes as other titles
title_mapping = {"Mr": 1, "Miss": 2, "Ms": 2, "Mrs": 3,
				"Master": 4, "Dr": 5, "Rev": 6,
				"Major": 7, "Col": 7, "Capt": 7,
				"Mlle": 8, "Mme": 8, "Don": 9, "Sir": 9,
				"Lady": 10, "Countess": 10, "Jonkheer": 10}

for k,v in title_mapping.items():
    titles[titles == k] = v

# check that everything is converted
print(pandas.value_counts(titles))

# add in title column
titanic["Title"] = titles

# FAMILY SIZE -------------------------------------------------------------------------

# generate family size column
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

# .apply method generates a new series
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))

# FAMILY ID -------------------------------------------------------------------------

import operator

# dictionary mapping family name to id
family_id_mapping = {}

# function to get id given a row
def get_family_id(row):
    # find last name by splitting on a comma
    last_name = row["Name"].split(",")[0]
    # create family id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # look up id in mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # get maximum id from the mapping and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

# get family ids with apply method
family_ids = titanic.apply(get_family_id, axis=1)

# compress all families under 3 members into one code
family_ids[titanic["FamilySize"] < 3] = -1

# print count of each unique id
print(pandas.value_counts(family_ids))

titanic["FamilyId"] = family_ids

# SELECTING BEST VARIABLES -------------------------------------------------------------------------

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from matplotlib import pyplot as plt

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

# perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

# get raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# ploy the scores
# "Pclass", "Sex", "Title", and "Fare" are best
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

# pick four best features
predictors = ["Pclass", "Sex", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print(scores.mean())

# SELECTING BEST VARIABLES -------------------------------------------------------------------------

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

# ensemble algorithms
# use more linear predictors for logistic regression
# use gradient gradient boosting classifier for everything else
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

# initialize cross validation folds
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    # make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # fit algorithm on training data.
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        # select and predict on test fold.  
        # .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # use a simple ensembling scheme: average the predictions to get the final classification
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # >.5 is a 1 prediction, <.5 is a 0 prediction
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

# put all  predictions together into one array
predictions = np.concatenate(predictions, axis=0)

# compute accuracy by comparing to training data
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)

# CLEANING TEST -------------------------------------------------------------------------

titanic_test = pandas.read_csv("Data/test.csv")

# set null ages to median
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

# set male and female to 0 and 1
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

# clean Embarked data
titanic_test["Embarked"] = titanic_test["Embarked"].fillna('S')
titanic_test.loc[titanic_test["Embarked"] == 'S', "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == 'C', "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == 'Q', "Embarked"] = 2

# clean Fare data
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

# add titles to test set
titles = titanic_test["Name"].apply(get_title)
# note Dona title is added since it is in test set but not in training
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
for k,v in title_mapping.items():
    titles[titles == k] = v
titanic_test["Title"] = titles
# check counts of each unique title
print(pandas.value_counts(titanic_test["Title"]))

# add family size column
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]

# add family ids
print(family_id_mapping)

family_ids = titanic_test.apply(get_family_id, axis=1)
family_ids[titanic_test["FamilySize"] < 3] = -1
titanic_test["FamilyId"] = family_ids

# name length
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))


# RUNNING TEST -------------------------------------------------------------------------

predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

full_predictions = []
for alg, predictors in algorithms:
    # fit algorithm using full training data
    alg.fit(titanic[predictors], titanic["Survived"])
    # predict using test dataset
    # convert all columns to floats to avoid an error
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

# gradient boosting classifier generates better predictions, weight it higher
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4

predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1

predictions = predictions.astype(int)

submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })

# create submission file
submission.to_csv("submission_randomtree.csv", index=False)

















