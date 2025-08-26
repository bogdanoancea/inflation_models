import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
import joblib


def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


df = pd.read_excel('/Users/bogdanoancea/OneDrive/papers/2024/Olomouc/GDP_Q2024.xlsx')

#print(df)

#df.rename(columns={'PRODUS INTERN BRUT': 'GDP'}, inplace=True)
df.set_index('Quarter', inplace=True)

train_start_dt = '1995Q1'
train_end_dt = '2022Q4'
test_start_dt = '2021Q3'

train = df.iloc[df.index<=train_end_dt,][['GDP']]
test = df.iloc[df.index>=test_start_dt,][['GDP']]
print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)

scaler = MinMaxScaler()
train['Inflation'] = scaler.fit_transform(train)
test['Inflation'] = scaler.transform(test)

# Converting to numpy arrays
train_data = train.values
test_data = test.values

timesteps=7

train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape

test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

tscv = TimeSeriesSplit(n_splits=3)

SVRm = True

if SVRm == True:
    modelSVR = SVR()
    param_gridSVR = {
        'C' : [1,2,3,4,5,6,7,8],
        'gamma' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'coef0' : [0.0, 0.5, 1.0, 2.0, 2.5],
        'epsilon' : [0.0, 0.01, 0.05, 0.1, 0.2, 0.3],
        'kernel' : ['rbf', 'poly'],
        'degree' : [1,2,3,4,5,6]
    }

    grid_searchSVR = GridSearchCV(modelSVR, param_gridSVR, cv=tscv, scoring='neg_mean_squared_error')
    # Fit GridSearchCV
    grid_searchSVR.fit(x_train, y_train[:,0])
    # Print the best parameters and best score for SVR
    print("Best parameters:", grid_searchSVR.best_params_)
    print("Best cross-validation score: {:.2f}".format(-grid_searchSVR.best_score_))
    grid_searchSVR.best_estimator_
    joblib_file = "SVR_GDPmodel_1series.pkl"
    joblib.dump(grid_searchSVR.best_estimator_, joblib_file)

    y_train_pred = grid_searchSVR.best_estimator_.predict(x_train).reshape(-1, 1)
    y_test_pred = grid_searchSVR.best_estimator_.predict(x_test).reshape(-1, 1)
else:
    modelRF = RandomForestRegressor(n_jobs=10, random_state=1)
    # Parameters to tune
    param_gridRF = {
        'n_estimators': [75, 100, 150, 200],
        'max_depth': [None, 5, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 12, 15, 20],
        'max_features': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    grid_searchRF = GridSearchCV(modelRF, param_gridRF, cv=tscv, scoring='neg_mean_squared_error')
    # Fit GridSearchCV
    grid_searchRF.fit(x_train, y_train[:, 0])
    # Print the best parameters and best score for RF
    print("Best parameters:", grid_searchRF.best_params_)
    print("Best cross-validation score: {:.2f}".format(-grid_searchRF.best_score_))

    grid_searchRF.best_estimator_
    joblib_file = "random_forest_model_GDP_1series.pkl"
    joblib.dump(grid_searchRF.best_estimator_, joblib_file)

    y_train_pred = grid_searchRF.best_estimator_.predict(x_train).reshape(-1, 1)
    y_test_pred = grid_searchRF.best_estimator_.predict(x_test).reshape(-1, 1)

print(y_train_pred.shape, y_test_pred.shape)

# Scaling the predictions
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)
# Scaling the original values
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

print(len(y_train), len(y_test))
print(len(y_train_pred), len(y_test_pred))

train_timestamps = df[(df.index <= train_end_dt) & (df.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = df[(df.index >train_end_dt)].index[0:]

print(len(train_timestamps), len(test_timestamps))

print('MAPE for training data: ', sklearn.metrics.mean_absolute_percentage_error(y_train, y_train_pred)*100, '%')
print('MSE for training data: ', sklearn.metrics.mean_squared_error(y_train, y_train_pred))
print('MAPE for testing data: ', sklearn.metrics.mean_absolute_percentage_error(y_test, y_test_pred)*100, '%')
print('MSE for testing data: ', sklearn.metrics.mean_squared_error(y_test, y_test_pred))


