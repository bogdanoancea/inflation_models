import pandas as pd
import openpyxl
import numpy as np
from numpy import array
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.svm import SVR
import sklearn.metrics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('MacOSX')
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler

def create_model(optimizer='adam', lstm_neurons=50, activation='relu', recurrent_dropout = 0.1, kernel_regularizer = regularizers.l2(0.01)):
    model = Sequential()
    model.add(LSTM(lstm_neurons, activation=activation, return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(lstm_neurons, activation=activation, recurrent_dropout=recurrent_dropout))
    model.add(Dense(n_steps_out, kernel_regularizer=kernel_regularizer))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
    return model

# split a multivariate sequence into samples
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



# multivariate - all three series
df = pd.read_excel('/Users/bogdanoancea/OneDrive/papers/2024/time-series/date.xlsx')
ts = df.iloc[:, [0, 1, 3, 4]]
ts.rename(columns={'quarter': 'Quarter', 'rata inflatiei': 'Inflation', 'indice sentiment': 'Sentiment','rata somajului': 'Unemployment'}, inplace=True)

columns_titles = ["Quarter", "Sentiment","Unemployment", "Inflation"]
ts=ts.reindex(columns=columns_titles)
ts.set_index('Quarter', inplace=True)

train_start_dt = '2006:Q1'
train_end_dt = '2022:Q4'
test_start_dt = '2021:Q3'

train = ts.iloc[ts.index<=train_end_dt,][['Inflation', 'Sentiment','Unemployment']]
test = ts.iloc[ts.index>=test_start_dt,][['Inflation', 'Sentiment','Unemployment']]
print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)

# Converting to numpy arrays
train_data = train.values
test_data = test.values

scale = False
if scale:
    scalerX = MinMaxScaler()
    train_data = scalerX.fit_transform(train_data)
    test_data = scalerX.fit_transform(test_data)

# choose a number of time steps
n_steps_in, n_steps_out = 6, 1
# covert into input/output
x_train, y_train = split_sequences(train_data, n_steps_in, n_steps_out)
x_test, y_test = split_sequences(test_data, n_steps_in, n_steps_out)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


n_features = x_train.shape[2]
# define model
# Keras model with SciKeras wrapper
model = KerasRegressor(model=create_model, shuffle=False, verbose=2)

# Hyperparameters to be optimized
param_grid2 = {
    'model__optimizer': ['adam'],      # Note the prefix "model__"
    'model__lstm_neurons': [500, 1000],         # Note the prefix "model__"
    'model__recurrent_dropout' : [0.1, 0.2, 0.3],   # Note the prefix "model__"
    'model__kernel_regularizer' : [regularizers.l2(0.00), regularizers.l2(0.01), regularizers.l2(0.02)],
    'batch_size': [1,4],
    'epochs': [500, 1000]
}

# GridSearchCV
tscv = TimeSeriesSplit(n_splits=3)

grid = GridSearchCV(estimator=model, param_grid=param_grid2, scoring='neg_mean_squared_error', cv=tscv, verbose=2, n_jobs= -1)
grid_result = grid.fit(x_train, y_train ,shuffle = False)

# Display the best hyperparameters
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# best model
model = grid_result.best_estimator_
model.model_.save("best_LSTM3series.keras")
model = create_model(optimizer='adam',
                     lstm_neurons=grid_result.best_params_['model__lstm_neurons'],
                     activation='relu',
                     recurrent_dropout = grid_result.best_params_['model__recurrent_dropout'],
                     kernel_regularizer = grid_result.best_params_['model__kernel_regularizer']
)
model.save('best_LSTM3series2.keras')

model = keras.models.load_model('best_LSTM3series2.keras')
history = model.fit(x_train, y_train,  epochs=grid_result.best_params_['epochs'], verbose=2, shuffle = False, batch_size=grid_result.best_params_['batch_size'])
history2 = model.fit(x_train, y_train,  epochs=1000, verbose=2, shuffle = False, batch_size=1)
model = grid_result.best_estimator_
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

if scale:
    # Scaling the predictions
    y_train_pred = scalerX.inverse_transform(y_train_pred)
    y_test_pred = scalerX.inverse_transform(y_test_pred)
    # Scaling the original values
    y_train = scalerX.inverse_transform(y_train)
    y_test = scalerX.inverse_transform(y_test)

print(len(y_train), len(y_test))
print(len(y_train_pred), len(y_test_pred))

timesteps=7
train_timestamps = ts[(ts.index <= train_end_dt) & (ts.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = ts[(ts.index >train_end_dt)].index[0:]

print(len(train_timestamps), len(test_timestamps))
print('MAPE for training data: ', sklearn.metrics.mean_absolute_percentage_error(y_train, y_train_pred)*100, '%')
print('MSE for training data: ', sklearn.metrics.mean_squared_error(y_train, y_train_pred))
print('MAPE for testing data: ', sklearn.metrics.mean_absolute_percentage_error(y_test, y_test_pred)*100, '%')
print('MSE for testing data: ', sklearn.metrics.mean_squared_error(y_test, y_test_pred))


s = plt.figure(figsize=(16,10))
plt.rcParams.update({'font.size': 13})
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.08)
plt.subplots_adjust(top=0.95)
plt.subplots_adjust(right=0.95)
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=1)
plt.plot(test_timestamps, y_test, color = 'purple', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'navy', linewidth=1)
plt.legend(['Train Actual','Train Predicted', 'Test Actual', 'Test Predicted'])
plt.xticks(rotation=90)
plt.xlabel('Timestamp')
plt.ylabel('Inflation (%)')
plt.title("LSTM: Training/Testing actual vs. predicted values.")
s.savefig('LSTM-MULTIvar.eps', format='eps', dpi=1200)
plt.show()


#Create a figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10))
plt.rcParams.update({'font.size': 13})
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.08)
plt.subplots_adjust(top=0.95)
plt.subplots_adjust(right=0.95)

# Plot on the first subplot
ax1.plot(history.history['loss'], linewidth=2.0, alpha = 0.6, label = 'MSE')
ax1.set_title('MSE')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('MSE')
ax1.legend()

# Plot on the second subplot
ax2.plot(history.history['mae'], linewidth=2.0, alpha = 0.6, label = 'MAE')
ax2.set_title('MAE')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('MAE')
ax2.legend()

# Plot on the third subplot
ax3.plot(history.history['mape'], linewidth=2.0, alpha = 0.6, label = 'MAPE')
ax3.set_title('MAPE')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('MAPE')
ax3.legend()
# Adjust layout
plt.tight_layout()

# Save the figure if needed
# plt.savefig('trig_functions.png')
plt.savefig('LSTM-MULTIvar-Metrics.eps', format='eps', dpi=1200)
# Show the plot
plt.show()




#acum cu datele mele serie univariata
df = pd.read_excel('/Users/bogdanoancea/OneDrive/papers/2024/time-series/date.xlsx')
ts = df.iloc[:, [0, 1, 3, 4]]
ts.rename(columns={'quarter': 'Quarter', 'rata inflatiei': 'Inflation', 'indice sentiment': 'Sentiment','rata somajului': 'Unemployment'}, inplace=True)

columns_titles = ["Quarter", "Sentiment","Unemployment", "Inflation"]
ts=ts.reindex(columns=columns_titles)
ts.set_index('Quarter', inplace=True)

train_start_dt = '2006:Q1'
train_end_dt = '2022:Q4'
test_start_dt = '2021:Q3'

train = ts.iloc[ts.index<=train_end_dt,][['Inflation']]
test = ts.iloc[ts.index>=test_start_dt,][['Inflation']]
print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)

# Converting to numpy arrays
train_data = train.values
test_data = test.values

scale = False
if scale:
    scalerX = MinMaxScaler()
    train_data = scalerX.fit_transform(train_data)
    test_data = scalerX.fit_transform(test_data)

timesteps=7


# choose a number of time steps
n_steps_in, n_steps_out = 6, 1
# covert into input/output
x_train, y_train = split_sequences(train_data, n_steps_in, n_steps_out)
x_test, y_test = split_sequences(test_data, n_steps_in, n_steps_out)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

n_features = x_train.shape[2]
# define model
# Keras model with SciKeras wrapper
model = KerasRegressor(model=create_model, shuffle=False, verbose=2)

# Hyperparameters to be optimized
param_grid2 = {
    'model__optimizer': ['adam'],      # Note the prefix "model__"
    'model__lstm_neurons': [500, 1000],         # Note the prefix "model__"
    'model__recurrent_dropout' : [0.1, 0.2, 0.3],   # Note the prefix "model__"
    'model__kernel_regularizer' : [regularizers.l2(0.00), regularizers.l2(0.01), regularizers.l2(0.02)],
    'batch_size': [1,4],
    'epochs': [500, 1000]
}

# GridSearchCV
tscv = TimeSeriesSplit(n_splits=3)

grid = GridSearchCV(estimator=model, param_grid=param_grid2, scoring='neg_mean_squared_error', cv=tscv, verbose=2, n_jobs= -1)
grid_result = grid.fit(x_train, y_train ,shuffle = False)

# Display the best hyperparameters
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# best model
model = grid_result.best_estimator_
model.model_.save("best_LSTMuni.keras")
model = create_model(optimizer='adam',
                     lstm_neurons=grid_result.best_params_['model__lstm_neurons'],
                     activation='relu',
                     recurrent_dropout = grid_result.best_params_['model__recurrent_dropout'],
                     kernel_regularizer = grid_result.best_params_['model__kernel_regularizer']
)
model.save('best_LSTMuni2.keras')
#model = keras.saving.load_model('best_LSTMuni2.keras')

#model = keras.models.load_model('best_model.keras')
history = model.fit(x_train, y_train,  epochs=grid_result.best_params_['epochs'], verbose=2, shuffle = False, batch_size=grid_result.best_params_['batch_size'])
history = model.fit(x_train, y_train,  epochs=1000, verbose=1, shuffle = False, batch_size=1)
model = grid_result.best_estimator_
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

if scale:
    # Scaling the predictions
    y_train_pred = scalerX.inverse_transform(y_train_pred)
    y_test_pred = scalerX.inverse_transform(y_test_pred)
    # Scaling the original values
    y_train = scalerX.inverse_transform(y_train)
    y_test = scalerX.inverse_transform(y_test)

print(len(y_train), len(y_test))
print(len(y_train_pred), len(y_test_pred))

timesteps=7
train_timestamps = ts[(ts.index <= train_end_dt) & (ts.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = ts[(ts.index >train_end_dt)].index[0:]

print(len(train_timestamps), len(test_timestamps))
print('MAPE for training data: ', sklearn.metrics.mean_absolute_percentage_error(y_train, y_train_pred)*100, '%')
print('MSE for training data: ', sklearn.metrics.mean_squared_error(y_train, y_train_pred))
print('MAPE for testing data: ', sklearn.metrics.mean_absolute_percentage_error(y_test, y_test_pred)*100, '%')
print('MSE for testing data: ', sklearn.metrics.mean_squared_error(y_test, y_test_pred))


s = plt.figure(figsize=(16,10))
plt.rcParams.update({'font.size': 13})
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.08)
plt.subplots_adjust(top=0.95)
plt.subplots_adjust(right=0.95)
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=1)
plt.plot(test_timestamps, y_test, color = 'purple', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'navy', linewidth=1)
plt.legend(['Train Actual','Train Predicted', 'Test Actual', 'Test Predicted'])
plt.xticks(rotation=90)
plt.xlabel('Timestamp')
plt.ylabel('Inflation (%)')
plt.title("LSTM: Training/Testing actual vs. predicted values.")
s.savefig('LSTM-univarvar.eps', format='eps', dpi=1200)
plt.show()


#Create a figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10))
plt.rcParams.update({'font.size': 13})
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.08)
plt.subplots_adjust(top=0.95)
plt.subplots_adjust(right=0.95)

# Plot on the first subplot
ax1.plot(history.history['loss'], linewidth=2.0, alpha = 0.6, label = 'MSE')
ax1.set_title('MSE')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('MSE')
ax1.legend()

# Plot on the second subplot
ax2.plot(history.history['mae'], linewidth=2.0, alpha = 0.6, label = 'MAE')
ax2.set_title('MAE')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('MAE')
ax2.legend()

# Plot on the third subplot
ax3.plot(history.history['mape'], linewidth=2.0, alpha = 0.6, label = 'MAPE')
ax3.set_title('MAPE')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('MAPE')
ax3.legend()
# Adjust layout
plt.tight_layout()

# Save the figure if needed
# plt.savefig('trig_functions.png')
plt.savefig('LSTM-univar-Metrics.eps', format='eps', dpi=1200)
# Show the plot
plt.show()







#acum doar cu unemployment
df = pd.read_excel('/Users/bogdanoancea/OneDrive/papers/2024/time-series/date.xlsx')
ts = df.iloc[:, [0, 1, 4]]
ts.rename(columns={'quarter': 'Quarter', 'rata inflatiei': 'Inflation','rata somajului': 'Unemployment'}, inplace=True)

columns_titles = ["Quarter", "Unemployment", "Inflation"]
ts=ts.reindex(columns=columns_titles)
ts.set_index('Quarter', inplace=True)


train_start_dt = '2006:Q1'
train_end_dt = '2022:Q4'
test_start_dt = '2021:Q3'


train = ts.iloc[ts.index<=train_end_dt,][['Inflation', 'Unemployment']]
test = ts.iloc[ts.index>=test_start_dt,][['Inflation', 'Unemployment']]
print('Training data shape: ', train.shape)
print('Test data shape: ', test.shape)

# Converting to numpy arrays
train_data = train.values
test_data = test.values

timesteps=7

# Converting to numpy arrays
train_data = train.values
test_data = test.values

scale = False
if scale:
    scalerX = MinMaxScaler()
    train_data = scalerX.fit_transform(train_data)
    test_data = scalerX.fit_transform(test_data)

# choose a number of time steps
n_steps_in, n_steps_out = 6, 1
# covert into input/output
x_train, y_train = split_sequences(train_data, n_steps_in, n_steps_out)
x_test, y_test = split_sequences(test_data, n_steps_in, n_steps_out)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


n_features = x_train.shape[2]
# define model
# Keras model with SciKeras wrapper
model = KerasRegressor(model=create_model, shuffle=False, verbose=2)

# Hyperparameters to be optimized
param_grid2 = {
    'model__optimizer': ['adam'],      # Note the prefix "model__"
    'model__lstm_neurons': [500, 1000],         # Note the prefix "model__"
    'model__recurrent_dropout' : [0.1, 0.2, 0.3],   # Note the prefix "model__"
    'model__kernel_regularizer' : [regularizers.l2(0.00), regularizers.l2(0.01), regularizers.l2(0.02)],
    'batch_size': [1,4],
    'epochs': [500, 1000]
}

# GridSearchCV
tscv = TimeSeriesSplit(n_splits=3)

grid = GridSearchCV(estimator=model, param_grid=param_grid2, scoring='neg_mean_squared_error', cv=tscv, verbose=2, n_jobs= -1)
grid_result = grid.fit(x_train, y_train ,shuffle = False)

# Display the best hyperparameters
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# best model
model = grid_result.best_estimator_
model.model_.save("best_LSTM2seriesInfl-Unempl2.keras")
model = create_model(optimizer='adam',
                     lstm_neurons=grid_result.best_params_['model__lstm_neurons'],
                     activation='relu',
                     recurrent_dropout = grid_result.best_params_['model__recurrent_dropout'],
                     kernel_regularizer = grid_result.best_params_['model__kernel_regularizer']
)
model.save('best_LSTM2seriesInfl-Unempl.keras')

model = keras.models.load_model('best_LSTM3series2.keras')
history = model.fit(x_train, y_train,  epochs=grid_result.best_params_['epochs'], verbose=2, shuffle = False, batch_size=grid_result.best_params_['batch_size'])
#history2 = model.fit(x_train, y_train,  epochs=1000, verbose=2, shuffle = False, batch_size=1)
model = grid_result.best_estimator_
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

if scale:
    # Scaling the predictions
    y_train_pred = scalerX.inverse_transform(y_train_pred)
    y_test_pred = scalerX.inverse_transform(y_test_pred)
    # Scaling the original values
    y_train = scalerX.inverse_transform(y_train)
    y_test = scalerX.inverse_transform(y_test)

print(len(y_train), len(y_test))
print(len(y_train_pred), len(y_test_pred))

timesteps=7
train_timestamps = ts[(ts.index <= train_end_dt) & (ts.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = ts[(ts.index >train_end_dt)].index[0:]

print(len(train_timestamps), len(test_timestamps))
print('MAPE for training data: ', sklearn.metrics.mean_absolute_percentage_error(y_train, y_train_pred)*100, '%')
print('MSE for training data: ', sklearn.metrics.mean_squared_error(y_train, y_train_pred))
print('MAPE for testing data: ', sklearn.metrics.mean_absolute_percentage_error(y_test, y_test_pred)*100, '%')
print('MSE for testing data: ', sklearn.metrics.mean_squared_error(y_test, y_test_pred))


s = plt.figure(figsize=(16,10))
plt.rcParams.update({'font.size': 13})
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.08)
plt.subplots_adjust(top=0.95)
plt.subplots_adjust(right=0.95)
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=1)
plt.plot(test_timestamps, y_test, color = 'purple', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'navy', linewidth=1)
plt.legend(['Train Actual','Train Predicted', 'Test Actual', 'Test Predicted'])
plt.xticks(rotation=90)
plt.xlabel('Timestamp')
plt.ylabel('Inflation (%)')
plt.title("LSTM: Training/Testing actual vs. predicted values.")
s.savefig('LSTM-MULTIvarInfl-Unempl.eps', format='eps', dpi=1200)
plt.show()


#Create a figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10))
plt.rcParams.update({'font.size': 13})
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.08)
plt.subplots_adjust(top=0.95)
plt.subplots_adjust(right=0.95)

# Plot on the first subplot
ax1.plot(history.history['loss'], linewidth=2.0, alpha = 0.6, label = 'MSE')
ax1.set_title('MSE')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('MSE')
ax1.legend()

# Plot on the second subplot
ax2.plot(history.history['mae'], linewidth=2.0, alpha = 0.6, label = 'MAE')
ax2.set_title('MAE')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('MAE')
ax2.legend()

# Plot on the third subplot
ax3.plot(history.history['mape'], linewidth=2.0, alpha = 0.6, label = 'MAPE')
ax3.set_title('MAPE')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('MAPE')
ax3.legend()
# Adjust layout
plt.tight_layout()

# Save the figure if needed
# plt.savefig('trig_functions.png')
plt.savefig('LSTM-MULTIvarInfl-Unempl-Metrics.eps', format='eps', dpi=1200)
# Show the plot
plt.show()

