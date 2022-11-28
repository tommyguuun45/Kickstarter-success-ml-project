import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')
import glob
import os
from sklearn.model_selection import GridSearchCV

RSEED = 42

from data_cleaning_feature_engineering import extract_dict_item, drop_column, filter_transform_target, round_values, make_encode

#kickstarter data
path = r'data/data-2' # use your path
all_files = glob.glob(os.path.join(path , "*.csv"))

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
df = pd.concat(li, axis=0, ignore_index=True)

#execute functions from data_cleaning and feature_engineering
df = extract_dict_item(df)
df = drop_column(df)
df = filter_transform_target(df)
df = round_values(df)
df = make_encode(df)

#preparing data and define target and feature
X = df.drop('state', axis= 1)
y = df['state']

#splitting into train and test
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, stratify = y, random_state=RSEED)
#We stratify the test split because we have an imbalance in the state column of the dataset
## in order to exemplify how the predict will work.. we will save the y_train




# model
print("Training XGBoost Classifier")
boost = xgb.XGBClassifier(random_state=RSEED)

#to tune our models we have find best parameters with GridSearchCV 
grid = GridSearchCV(estimator=boost,
        param_grid={"base_score": [0.3,0.5,0.6],
        'learning_rate': [0.1, 1, 10],
        'max_depth': [2, 4, 6, 8],
        'n_estimators': [50, 100, 200]},
        scoring="f1",
        cv=3)
        
grid.fit(X_train, y_train)
model_xgboost = grid.best_estimator_

print(f"Best model has the following params:{grid.best_params_} and the best estimator is:{grid.best_estimator_}")


y_pred_boost_train = model_xgboost.predict(X_train)
y_pred_boost_test = model_xgboost.predict(X_test)
y_pred_test_proba = model_xgboost.predict_proba(X_test)

# Conducting error analysis
# building confusion matrix
cfm_train = confusion_matrix(y_train,y_pred_boost_train)
cfm_test = confusion_matrix(y_test,y_pred_boost_test)

print ('Confusion Matrix of train_data of XGBoost:', cfm_train)
print ('Confusion Matrix of test_data of XGBoost:', cfm_test)
#saving the model
print("Saving model in the model folder")
filename_model = 'models/model_xgboost.sav'
pickle.dump(model_xgboost, open(filename_model, 'wb'))

print("Saving test data in the data folder")
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
pd.DataFrame(y_pred_boost_test).to_csv("data/y_pred_boost_test.csv", index=False)
pd.DataFrame(y_pred_test_proba).to_csv("data/y_pred_test_proba.csv", index=False)
