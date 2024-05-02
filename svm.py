import matplotlib.pyplot as plt

import pandas as pd
import sklearn.metrics

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

dt=pd.read_csv('/Users/bogdanoancea/OneDrive/papers/2024/time-series/energy.csv', index_col='timestamp')
energy = dt[['load']]

energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('load', fontsize=12)
plt.show()

train_start_dt = '2014-11-01 00:00:00'
test_start_dt = '2014-12-30 00:00:00'
#dt.set_index('timestamp')
energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}).join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer').plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('load', fontsize=12)
plt.show()

train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
test = energy.copy()[energy.index >= test_start_dt][['load']]

print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)

scaler = MinMaxScaler()
train['load'] = scaler.fit_transform(train)
test['load'] = scaler.transform(test)

# Converting to numpy arrays
train_data = train.values
test_data = test.values

timesteps=5
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape

test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape

x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)

model.fit(x_train, y_train[:,0])
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)
print(y_train_pred.shape, y_test_pred.shape)
# Scaling the predictions
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)
# Scaling the original values
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

print(len(y_train), len(y_test))
print(len(y_train_pred), len(y_test_pred))

train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
print('MAPE for training data: ', sklearn.metrics.mean_absolute_percentage_error(y_train_pred, y_train)*100, '%')
print('MSE for training data: ', sklearn.metrics.mean_squared_error(y_train_pred, y_train))

plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()

print('MAPE for testing data: ', sklearn.metrics.mean_absolute_percentage_error(y_test_pred, y_test)*100, '%')
print('MSE for testing data: ', sklearn.metrics.mean_squared_error(y_test_pred, y_test))
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()





#acum cu datele mele serie univariata
df = pd.read_excel('/Users/bogdanoancea/OneDrive/papers/2024/time-series/date.xlsx')
ts = df.iloc[:, [0, 1, 3, 4]]
ts.rename(columns={'quarter': 'Quarter', 'rata inflatiei': 'Inflation', 'indice sentiment': 'Sentiment','rata somajului': 'Unemployment'}, inplace=True)

columns_titles = ["Quarter", "Sentiment","Unemployment", "Inflation"]
ts=ts.reindex(columns=columns_titles)
ts.set_index('Quarter', inplace=True)

ts.plot( y='Inflation', subplots=True, figsize=(15, 8), fontsize=12)
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Inflation', fontsize=12)
plt.show()

train_start_dt = '2006:Q1'
train_end_dt = '2022:Q4'
test_start_dt = '2021:Q3'

ts.iloc[ (ts.index <= train_end_dt) & (ts.index >= train_start_dt)][['Inflation']].rename(columns={'Inflation':'train'}).join(ts.iloc[ts.index>train_end_dt][['Inflation']].rename(columns={'Inflation':'test'}), how='outer').plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Inflation', fontsize=12)
plt.show()

train = ts.iloc[ts.index<=train_end_dt,][['Inflation']]
test = ts.iloc[ts.index>=test_start_dt,][['Inflation']]
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
tscv = TimeSeriesSplit(n_splits=4)

param_grid = {
    'C': [1,2,3,4,5,6,7,8,9,10],
    'gamma': [0,1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'coef0': [0.0, 0.5, 1.0, 2.0],
    'epsilon' : [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]
}

model = SVR()
model = RandomForestRegressor()
# Parameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [1, 2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
# Fit GridSearchCV
grid_search.fit(x_train, y_train[:,0])

# Print the best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(-grid_search.best_score_))

grid_search.best_estimator_
#model.fit(x_train, y_train[:,0])
y_train_pred = grid_search.best_estimator_.predict(x_train).reshape(-1,1)
y_test_pred = grid_search.best_estimator_.predict(x_test).reshape(-1,1)
print(y_train_pred.shape, y_test_pred.shape)
# Scaling the predictions
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)
# Scaling the original values
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

print(len(y_train), len(y_test))
print(len(y_train_pred), len(y_test_pred))

train_timestamps = ts[(ts.index <= train_end_dt) & (ts.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = ts[(ts.index >train_end_dt)].index[0:]

print(len(train_timestamps), len(test_timestamps))
print('MAPE for training data: ', sklearn.metrics.mean_absolute_percentage_error(y_train_pred, y_train)*100, '%')
print('MSE for training data: ', sklearn.metrics.mean_squared_error(y_train_pred, y_train))


plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()

print('MAPE for testing data: ', sklearn.metrics.mean_absolute_percentage_error(y_test_pred, y_test)*100, '%')
print('MSE for testing data: ', sklearn.metrics.mean_squared_error(y_test_pred, y_test))
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()



#acum cu datele mele multivariate
df = pd.read_excel('/Users/bogdanoancea/OneDrive/papers/2024/time-series/date.xlsx')
ts = df.iloc[:, [0, 1, 3, 4]]
ts.rename(columns={'quarter': 'Quarter', 'rata inflatiei': 'Inflation', 'indice sentiment': 'Sentiment','rata somajului': 'Unemployment'}, inplace=True)

columns_titles = ["Quarter", "Sentiment","Unemployment", "Inflation"]
ts=ts.reindex(columns=columns_titles)
ts.set_index('Quarter', inplace=True)

ts.plot( y='Inflation', subplots=True, figsize=(15, 8), fontsize=12)
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Inflation', fontsize=12)
plt.show()

train_start_dt = '2006:Q1'
train_end_dt = '2022:Q4'
test_start_dt = '2021:Q3'

ts.iloc[ (ts.index <= train_end_dt) & (ts.index >= train_start_dt)][['Inflation']].rename(columns={'Inflation':'train'}).join(ts.iloc[ts.index>train_end_dt][['Inflation']].rename(columns={'Inflation':'test'}), how='outer').plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Inflation', fontsize=12)
plt.show()

train = ts.iloc[ts.index<=train_end_dt,][['Inflation', 'Sentiment','Unemployment']]
test = ts.iloc[ts.index>=test_start_dt,][['Inflation', 'Sentiment','Unemployment']]
print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)

#scaler = MinMaxScaler()
#train = scaler.fit_transform(train)
#test = scaler.transform(test)

# Converting to numpy arrays
train_data = train.values
test_data = test.values

timesteps=7

train_data_timestepsInfl=np.array([[j for j in train_data[i:i+timesteps-1,0]] for i in range(0,len(train_data)-timesteps+1)])[:,:]
train_data_timestepsSent=np.array([[j for j in train_data[i:i+timesteps-1,1]] for i in range(0,len(train_data)-timesteps+1)])[:,:]
train_data_timestepsUnempl=np.array([[j for j in train_data[i:i+timesteps-1,2]] for i in range(0,len(train_data)-timesteps+1)])[:,:]
train_data_timestepsTarget = np.array([[j for j in train_data[[i+timesteps-1],0]] for i in range(0,len(train_data)-timesteps+1)])[:,:]
train_data_timesteps = np.concatenate((train_data_timestepsInfl, train_data_timestepsSent, train_data_timestepsUnempl, train_data_timestepsTarget), axis=1)
train_data_timesteps.shape

test_data_timestepsInfl=np.array([[j for j in test_data[i:i+timesteps-1,0]] for i in range(0,len(test_data)-timesteps+1)])[:,:]
test_data_timestepsSent=np.array([[j for j in test_data[i:i+timesteps-1,1]] for i in range(0,len(test_data)-timesteps+1)])[:,:]
test_data_timestepsUnempl=np.array([[j for j in test_data[i:i+timesteps-1,2]] for i in range(0,len(test_data)-timesteps+1)])[:,:]
test_data_timestepsTarget=np.array([[j for j in test_data[[i+timesteps-1],0]] for i in range(0,len(test_data)-timesteps+1)])[:,:]

test_data_timesteps = np.concatenate((test_data_timestepsInfl, test_data_timestepsSent, test_data_timestepsUnempl,test_data_timestepsTarget), axis=1)
test_data_timesteps.shape

scalerX = MinMaxScaler()
scalerY = MinMaxScaler()
x_train, y_train = train_data_timesteps[:,:np.shape(train_data_timesteps)[1]-1],train_data_timesteps[:,[np.shape(train_data_timesteps)[1]-1]]
x_test, y_test = test_data_timesteps[:,:np.shape(train_data_timesteps)[1]-1],test_data_timesteps[:,[np.shape(train_data_timesteps)[1]-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = scalerX.fit_transform(x_train)
y_train = scalerY.fit_transform(y_train)
x_test = scalerX.transform(x_test)
y_test = scalerY.transform(y_test)

tscv = TimeSeriesSplit(n_splits=4)

param_grid = {
    'C': [1,2,3,4,5,6,7,8,9,10],
    'gamma': [0,1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'coef0': [0.0, 0.5, 1.0, 2.0],
    'epsilon' : [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]
}

model = SVR()
model = RandomForestRegressor()
# Parameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [1, 2, 5, 10]
}
# prepare the model with input scaling
#pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', SVR(kernel='rbf'))])
# prepare the model with target scaling
#wrapped_model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(model, x_train, y_train[:,0], scoring='neg_mean_absolute_error', cv=tscv, n_jobs=-1)
grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
# Fit GridSearchCV

grid_search.fit(x_train, y_train[:,0])

# Print the best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(-grid_search.best_score_))

grid_search.best_estimator_
#model.fit(x_train, y_train[:,0])
y_train_pred = grid_search.best_estimator_.predict(x_train).reshape(-1,1)
y_test_pred = grid_search.best_estimator_.predict(x_test).reshape(-1,1)
print(y_train_pred.shape, y_test_pred.shape)
# Scaling the predictions
y_train_pred = scalerY.inverse_transform(y_train_pred)
y_test_pred = scalerY.inverse_transform(y_test_pred)
# Scaling the original values
y_train = scalerY.inverse_transform(y_train)
y_test = scalerY.inverse_transform(y_test)

print(len(y_train), len(y_test))
print(len(y_train_pred), len(y_test_pred))

train_timestamps = ts[(ts.index <= train_end_dt) & (ts.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = ts[(ts.index >train_end_dt)].index[0:]

print(len(train_timestamps), len(test_timestamps))
print('MAPE for training data: ', sklearn.metrics.mean_absolute_percentage_error(y_train_pred, y_train)*100, '%')
print('MSE for training data: ', sklearn.metrics.mean_squared_error(y_train_pred, y_train))


plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()

print('MAPE for testing data: ', sklearn.metrics.mean_absolute_percentage_error(y_test_pred, y_test)*100, '%')
print('MSE for testing data: ', sklearn.metrics.mean_squared_error(y_test_pred, y_test))
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()

#acum doar cu unemployment
df = pd.read_excel('/Users/bogdanoancea/OneDrive/papers/2024/time-series/date.xlsx')
ts = df.iloc[:, [0, 1, 4]]
ts.rename(columns={'quarter': 'Quarter', 'rata inflatiei': 'Inflation','rata somajului': 'Unemployment'}, inplace=True)

columns_titles = ["Quarter", "Unemployment", "Inflation"]
ts=ts.reindex(columns=columns_titles)
ts.set_index('Quarter', inplace=True)

ts.plot( y='Inflation', subplots=True, figsize=(15, 8), fontsize=12)
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Inflation', fontsize=12)
plt.show()

train_start_dt = '2006:Q1'
train_end_dt = '2022:Q4'
test_start_dt = '2021:Q3'

ts.iloc[ (ts.index <= train_end_dt) & (ts.index >= train_start_dt)][['Inflation']].rename(columns={'Inflation':'train'}).join(ts.iloc[ts.index>train_end_dt][['Inflation']].rename(columns={'Inflation':'test'}), how='outer').plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Inflation', fontsize=12)
plt.show()

train = ts.iloc[ts.index<=train_end_dt,][['Inflation', 'Unemployment']]
test = ts.iloc[ts.index>=test_start_dt,][['Inflation', 'Unemployment']]
print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)

# Converting to numpy arrays
train_data = train.values
test_data = test.values

timesteps=7

train_data_timestepsInfl=np.array([[j for j in train_data[i:i+timesteps-1,0]] for i in range(0,len(train_data)-timesteps+1)])[:,:]
train_data_timestepsUnempl=np.array([[j for j in train_data[i:i+timesteps-1,1]] for i in range(0,len(train_data)-timesteps+1)])[:,:]
train_data_timestepsTarget = np.array([[j for j in train_data[[i+timesteps-1],0]] for i in range(0,len(train_data)-timesteps+1)])[:,:]
train_data_timesteps = np.concatenate((train_data_timestepsInfl, train_data_timestepsUnempl, train_data_timestepsTarget), axis=1)
train_data_timesteps.shape

test_data_timestepsInfl=np.array([[j for j in test_data[i:i+timesteps-1,0]] for i in range(0,len(test_data)-timesteps+1)])[:,:]
test_data_timestepsUnempl=np.array([[j for j in test_data[i:i+timesteps-1,1]] for i in range(0,len(test_data)-timesteps+1)])[:,:]
test_data_timestepsTarget=np.array([[j for j in test_data[[i+timesteps-1],0]] for i in range(0,len(test_data)-timesteps+1)])[:,:]

test_data_timesteps = np.concatenate((test_data_timestepsInfl,test_data_timestepsUnempl,test_data_timestepsTarget), axis=1)
test_data_timesteps.shape

scalerX = MinMaxScaler()
scalerY = MinMaxScaler()
x_train, y_train = train_data_timesteps[:,:np.shape(train_data_timesteps)[1]-1],train_data_timesteps[:,[np.shape(train_data_timesteps)[1]-1]]
x_test, y_test = test_data_timesteps[:,:np.shape(train_data_timesteps)[1]-1],test_data_timesteps[:,[np.shape(train_data_timesteps)[1]-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = scalerX.fit_transform(x_train)
y_train = scalerY.fit_transform(y_train)
x_test = scalerX.transform(x_test)
y_test = scalerY.transform(y_test)

tscv = TimeSeriesSplit(n_splits=4)

param_grid = {
    'C': [1,2,3,4,5,6,7,8,9,10],
    'gamma': [0,1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'coef0': [0.0, 0.5, 1.0, 2.0],
    'epsilon' : [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]
}

model = SVR()
model = RandomForestRegressor()
# Parameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [1, 2, 5, 10]
}
# prepare the model with input scaling
#pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', SVR(kernel='rbf'))])
# prepare the model with target scaling
#wrapped_model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(model, x_train, y_train[:,0], scoring='neg_mean_absolute_error', cv=tscv, n_jobs=-1)
grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
# Fit GridSearchCV

grid_search.fit(x_train, y_train[:,0])

# Print the best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(-grid_search.best_score_))

grid_search.best_estimator_
#model.fit(x_train, y_train[:,0])
y_train_pred = grid_search.best_estimator_.predict(x_train).reshape(-1,1)
y_test_pred = grid_search.best_estimator_.predict(x_test).reshape(-1,1)
print(y_train_pred.shape, y_test_pred.shape)
# Scaling the predictions
y_train_pred = scalerY.inverse_transform(y_train_pred)
y_test_pred = scalerY.inverse_transform(y_test_pred)
# Scaling the original values
y_train = scalerY.inverse_transform(y_train)
y_test = scalerY.inverse_transform(y_test)

print(len(y_train), len(y_test))
print(len(y_train_pred), len(y_test_pred))

train_timestamps = ts[(ts.index <= train_end_dt) & (ts.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = ts[(ts.index >train_end_dt)].index[0:]

print(len(train_timestamps), len(test_timestamps))
print('MAPE for training data: ', sklearn.metrics.mean_absolute_percentage_error(y_train_pred, y_train)*100, '%')
print('MSE for training data: ', sklearn.metrics.mean_squared_error(y_train_pred, y_train))


plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()

print('MAPE for testing data: ', sklearn.metrics.mean_absolute_percentage_error(y_test_pred, y_test)*100, '%')
print('MSE for testing data: ', sklearn.metrics.mean_squared_error(y_test_pred, y_test))
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()

#acum doar cu sentiment
df = pd.read_excel('/Users/bogdanoancea/OneDrive/papers/2024/time-series/date.xlsx')
ts = df.iloc[:, [0, 1, 3]]
ts.rename(columns={'quarter': 'Quarter', 'rata inflatiei': 'Inflation', 'indice sentiment': 'Sentiment'}, inplace=True)

columns_titles = ["Quarter", "Sentiment", "Inflation"]
ts=ts.reindex(columns=columns_titles)
ts.set_index('Quarter', inplace=True)

ts.plot( y='Inflation', subplots=True, figsize=(15, 8), fontsize=12)
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Inflation', fontsize=12)
plt.show()

train_start_dt = '2006:Q1'
train_end_dt = '2022:Q4'
test_start_dt = '2021:Q3'

ts.iloc[ (ts.index <= train_end_dt) & (ts.index >= train_start_dt)][['Inflation']].rename(columns={'Inflation':'train'}).join(ts.iloc[ts.index>train_end_dt][['Inflation']].rename(columns={'Inflation':'test'}), how='outer').plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Inflation', fontsize=12)
plt.show()

train = ts.iloc[ts.index<=train_end_dt,][['Inflation', 'Sentiment','Unemployment']]
test = ts.iloc[ts.index>=test_start_dt,][['Inflation', 'Sentiment','Unemployment']]
print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)

#scaler = MinMaxScaler()
#train = scaler.fit_transform(train)
#test = scaler.transform(test)

# Converting to numpy arrays
train_data = train.values
test_data = test.values

timesteps=7

train_data_timestepsInfl=np.array([[j for j in train_data[i:i+timesteps-1,0]] for i in range(0,len(train_data)-timesteps+1)])[:,:]
train_data_timestepsSent=np.array([[j for j in train_data[i:i+timesteps-1,1]] for i in range(0,len(train_data)-timesteps+1)])[:,:]
train_data_timestepsTarget = np.array([[j for j in train_data[[i+timesteps-1],0]] for i in range(0,len(train_data)-timesteps+1)])[:,:]
train_data_timesteps = np.concatenate((train_data_timestepsInfl, train_data_timestepsSent, train_data_timestepsUnempl, train_data_timestepsTarget), axis=1)
train_data_timesteps.shape

test_data_timestepsInfl=np.array([[j for j in test_data[i:i+timesteps-1,0]] for i in range(0,len(test_data)-timesteps+1)])[:,:]
test_data_timestepsSent=np.array([[j for j in test_data[i:i+timesteps-1,1]] for i in range(0,len(test_data)-timesteps+1)])[:,:]
test_data_timestepsTarget=np.array([[j for j in test_data[[i+timesteps-1],0]] for i in range(0,len(test_data)-timesteps+1)])[:,:]

test_data_timesteps = np.concatenate((test_data_timestepsInfl, test_data_timestepsSent, test_data_timestepsUnempl,test_data_timestepsTarget), axis=1)
test_data_timesteps.shape

scalerX = MinMaxScaler()
scalerY = MinMaxScaler()
x_train, y_train = train_data_timesteps[:,:np.shape(train_data_timesteps)[1]-1],train_data_timesteps[:,[np.shape(train_data_timesteps)[1]-1]]
x_test, y_test = test_data_timesteps[:,:np.shape(train_data_timesteps)[1]-1],test_data_timesteps[:,[np.shape(train_data_timesteps)[1]-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = scalerX.fit_transform(x_train)
y_train = scalerY.fit_transform(y_train)
x_test = scalerX.transform(x_test)
y_test = scalerY.transform(y_test)

tscv = TimeSeriesSplit(n_splits=4)

param_grid = {
    'C': [1,2,3,4,5,6,7,8,9,10],
    'gamma': [0,1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'coef0': [0.0, 0.5, 1.0, 2.0],
    'epsilon' : [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]
}

model = SVR()
model = RandomForestRegressor()
# Parameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [1, 2, 5, 10]
}
# prepare the model with input scaling
#pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', SVR(kernel='rbf'))])
# prepare the model with target scaling
#wrapped_model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(model, x_train, y_train[:,0], scoring='neg_mean_absolute_error', cv=tscv, n_jobs=-1)
grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
# Fit GridSearchCV

grid_search.fit(x_train, y_train[:,0])

# Print the best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(-grid_search.best_score_))

grid_search.best_estimator_
#model.fit(x_train, y_train[:,0])
y_train_pred = grid_search.best_estimator_.predict(x_train).reshape(-1,1)
y_test_pred = grid_search.best_estimator_.predict(x_test).reshape(-1,1)
print(y_train_pred.shape, y_test_pred.shape)
# Scaling the predictions
y_train_pred = scalerY.inverse_transform(y_train_pred)
y_test_pred = scalerY.inverse_transform(y_test_pred)
# Scaling the original values
y_train = scalerY.inverse_transform(y_train)
y_test = scalerY.inverse_transform(y_test)

print(len(y_train), len(y_test))
print(len(y_train_pred), len(y_test_pred))

train_timestamps = ts[(ts.index <= train_end_dt) & (ts.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = ts[(ts.index >train_end_dt)].index[0:]

print(len(train_timestamps), len(test_timestamps))
print('MAPE for training data: ', sklearn.metrics.mean_absolute_percentage_error(y_train_pred, y_train)*100, '%')
print('MSE for training data: ', sklearn.metrics.mean_squared_error(y_train_pred, y_train))


plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()

print('MAPE for testing data: ', sklearn.metrics.mean_absolute_percentage_error(y_test_pred, y_test)*100, '%')
print('MSE for testing data: ', sklearn.metrics.mean_squared_error(y_test_pred, y_test))
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()