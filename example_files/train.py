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
RSEED = 42

from data_cleaning_feature_engineering import extract_dict_item, drop_column, filter_transform_target, round_values, make_encode

#kickstarter data 
kickstarter = pd.read_csv('data/kickstarter_preprocessed.csv')

##feature eng on test data
#print("Feature engineering on train")
#X_train = transform_altitude(X_train)
#X_train = drop_column(X_train, col_name='Unnamed: 0')
#X_train = drop_column(X_train, col_name='Quakers')
#X_train = fill_missing_values(X_train)

#preparing data and define target and feature
X = kickstarter.drop('state', axis= 1)
y = kickstarter['state']
print('Our target is: state')


# splittin into train and test
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, stratify = y, random_state=RSEED)
'''We stratify the test split because we have an imbalance in the state column of the dataset'''
## in order to exemplify how the predict will work.. we will save the y_train
print("Saving test data in the data folder")
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

##feature eng on test data
#print("Feature engineering on train")
#X_train = transform_altitude(X_train)
#X_train = drop_column(X_train, col_name='Unnamed: 0')
#X_train = drop_column(X_train, col_name='Quakers')
#X_train = fill_missing_values(X_train)

# model
print("Training XGBoost Classifier")
boost = xgb.XGBClassifier( base_score = 0.5, learning_rate = 0.1, max_depth = 8, n_estimators = 200, random_state=RSEED)
model_xgboost= boost.fit(X_train,y_train)

#feature eng on test data
#print("Feature engineering on test")
#X_test = transform_altitude(X_test)
#X_test = drop_column(X_test, col_name='Unnamed: 0')
#X_test = drop_column(X_test, col_name='Quakers')
#X_test = fill_missing_values(X_test)

y_pred_boost = boost.predict(X_test)

# building confusion matrix
cfm_train = confusion_matrix(y_train,y_pred_boost)
cfm_test = confusion_matrix(y_test,y_pred_boost)

print ('Confusion Matrix of train_data of XGBoost:', cfm_train)
print ('Confusion Matrix of test_data of XGBoost:', cfm_test)
print("thats quite good")
#saving the model
print("Saving model in the model folder")
filename = 'models/model_xgboost.sav'
pickle.dump(boost, open(filename, 'wb'))