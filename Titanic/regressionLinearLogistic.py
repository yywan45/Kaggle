#! python3

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

# prints the unique categories in Embarked
print(titanic["Embarked"].unique())

# clean Embarked data
titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic.loc[titanic["Embarked"] == 'S', "Embarked"] = 0
titanic.loc[titanic["Embarked"] == 'C', "Embarked"] = 1
titanic.loc[titanic["Embarked"] == 'Q', "Embarked"] = 2

# print top 10 rows
print(titanic.head(30))

# print description of all data
print(titanic.describe())


# LINEAR REGRESSION ALGORITHM --------------------------------------------------------------------

# import linear regression class
from sklearn.linear_model import LinearRegression
# import helper function from sklearn for cross validation
from sklearn.cross_validation import KFold

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# initialize algorithm class
alg = LinearRegression()

# generate cross validation folds for the titanic dataset
# returns the row indices corresponding to train and test
# set random_state to ensure same splits every time this is run
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # predictors used to train algorithm
    # only take the rows in the train folds
    train_predictors = (titanic[predictors].iloc[train,:])
    # target used to train algorithm
    train_target = titanic["Survived"].iloc[train]
    # training
    alg.fit(train_predictors, train_target)
    # make predictions on test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

import numpy as np

# predictions are in three separate numpy arrays, concatenate into one
# concatenate them on axis 0, as they only have one axis
predictions = np.concatenate(predictions, axis=0)

# map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0

# find proportion of values in predictions that are same as "survived"
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)


# LOGISTIC REGRESSION ALGORITHM --------------------------------------------------------------------

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

# initialize algorithm
alg = LogisticRegression(random_state=1)

# compute accuracy score for all the cross validation folds
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

# take the mean of the scores (one for each fold)
print(scores.mean())


# USING THE TEST --------------------------------------------------------------------------------------------

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

# GENERATING SUBMISSION -------------------------------------------------------------------------------------

# initialize the algorithm class
alg = LogisticRegression(random_state=1)

# train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# make predictions using the test set
predictions = alg.predict(titanic_test[predictors])

# create a new dataframe with only the columns Kaggle wants from the dataset
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })

# create submission file
submission.to_csv("submission_regression.csv", index=False)






























