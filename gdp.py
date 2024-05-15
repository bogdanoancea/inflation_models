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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
from keras import regularizers

# split a univariate sequence into samples
def data_to_supervised(data, n_lags, n_out):
    X, y = list(), list()
    if n_out is None:
        for i in range(len(data)):
            # find the end of this pattern
            end_ix = i + n_lags
            # check if we are beyond the sequence
            if end_ix > len(data) - 1:
                break
            # gather input and output parts of the pattern
            xs, ys = data[i:end_ix], data[end_ix]
            X.append(xs)
            y.append(ys)
    else:
        for i in range(len(data)):
            end_ix = i + n_lags
            out_end_ix = end_ix + n_out
            # check if we are beyond the sequence
            if out_end_ix > len(data):
                break
            xs, ys = data[i:end_ix], data[end_ix:out_end_ix]
            X.append(xs)
            y.append(ys)

    return np.array(X), np.array(y)


df = pd.read_excel('/Users/bogdanoancea/OneDrive/papers/2024/Olomouc/GDP_Q2024.xlsx')

print(df)

df.rename(columns={'PRODUS INTERN BRUT': 'GDP'}, inplace=True)
df.set_index('Quarter', inplace=True)
# choose a number of time steps
n_steps_in, n_steps_out = 6, 1
# covert into input/output
X, y = data_to_supervised(df.to_numpy(), n_steps_in, None)
print(X.shape, y.shape)
# summarize the data
for i in range(len(X)):
	print(X[i], y[i])

n_features = X.shape[2]


# demonstrate prediction
def create_model(optimizer='adam', lstm_neurons=50, activation='relu', recurrent_dropout = 0.1, kernel_regularizer = regularizers.l2(0.01)):
    model = Sequential()
    model.add(LSTM(lstm_neurons, activation=activation, return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(lstm_neurons, activation=activation, recurrent_dropout=recurrent_dropout))
    model.add(Dense(n_steps_out, kernel_regularizer=kernel_regularizer))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
    return model

# Keras model with SciKeras wrapper
model = KerasRegressor(model=create_model, shuffle=False, verbose=2)

# Hyperparameters to be optimized
param_grid = {
    'model__optimizer': ['adam'],      # Note the prefix "model__"
    'model__lstm_neurons': [100, 500],         # Note the prefix "model__"
    'model__recurrent_dropout' : [0.0, 0.1, 0.2],   # Note the prefix "model__"
    'model__kernel_regularizer' : [regularizers.l2(0.0),  regularizers.l2(0.01), regularizers.l2(0.02)],
    'batch_size': [1,4,8],
    'epochs': [500]
}

param_grid2 = {
    'model__optimizer': ['adam'],      # Note the prefix "model__"
    'model__lstm_neurons': [500, 1000],         # Note the prefix "model__"
    'model__recurrent_dropout' : [0.1, 0.2, 0.3],   # Note the prefix "model__"
    'model__kernel_regularizer' : [regularizers.l2(0.03),  regularizers.l2(0.01), regularizers.l2(0.02)],
    'batch_size': [1,4,8],
    'epochs': [1000]
}

# GridSearchCV
tscv = TimeSeriesSplit(n_splits=3)

grid = GridSearchCV(estimator=model, param_grid=param_grid2, scoring='neg_mean_squared_error', cv=tscv, verbose=2, n_jobs= -1)
grid_result = grid.fit(X, y,shuffle = False)

# Display the best hyperparameters
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# define model
model = grid_result.best_estimator_
model.model_.save("best_model.keras")
model = keras.models.load_model('best_model.keras')
history = model.fit(X, y, epochs=grid_result.best_params_['epochs'], verbose=2, shuffle = False, batch_size=grid_result.best_params_['batch_size'], metrics=['mape', 'mae'])

plt.plot(history.history_['mse'])
plt.plot(history.history_['mean_absolute_percentage_error'])
plt.plot(history.history_['mape'])
plt.title('Model loss (mse), mae and mape for training data set')
plt.ylabel('mse')
plt.xlabel('Training epoch')
plt.show()



matplotlib.use('MacOSX')
plt.title('Performance metrics for training data set')
plt.plot(np.sqrt(history.history['loss']), label="Train RMSE")
plt.plot(history.history['mae'], label="Train MAE")
plt.xlabel("epochs")
plt.legend()
plt.show()

plt.figure()
plt.title('Performance metrics for training data set')
plt.plot(history.history['mape'], label="Train MAPE")
plt.xlabel("epochs")
plt.legend()
plt.show()

plt.plot(history.history['loss'])
plt.title('Model loss (mse) for training data set')
plt.ylabel('loss (mse)')
plt.xlabel('Training epoch')
plt.show()


plt.plot(y)
plt.plot(y_pred)
plt.show()

hist_df = pd.DataFrame(history.history)

# or save to csv:
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

mape = sklearn.metrics.mean_absolute_percentage_error(y[62,:], yhat[0])
mae = sklearn.metrics.mean_absolute_error(y[62], yhat[0])
mse = sklearn.metrics.mean_squared_error(y[62], yhat[0])
rmse = np.sqrt(mse)


yhat2 = model2.predict(x_input, verbose=0)
print(yhat2)

model.save("lstm.keras")
model2 = keras.models.load_model('lstm.keras')

np.save('my_history.npy',history.history)
history=np.load('my_history.npy',allow_pickle='TRUE').item()


matplotlib.use('MacOSX')
plt.plot(history.history['loss'])
plt.title('Model loss (mse) for training data set')
plt.ylabel('loss (mse)')
plt.xlabel('Training epoch')
plt.show()






from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import ConvLSTM2D
from matplotlib import pyplot
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle
from math import sqrt
import tensorflow as tf
import statistics as st
import gc
from multiprocessing import Process
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import RepeatVector


# split a univariate sequence into samples
def data_to_supervised(data, n_lags, n_out):
    X, y = list(), list()
    if n_out is None:
        for i in range(len(data)):
            # find the end of this pattern
            end_ix = i + n_lags
            # check if we are beyond the sequence
            if end_ix > len(data) - 1:
                break
            # gather input and output parts of the pattern
            xs, ys = data[i:end_ix], data[end_ix]
            X.append(xs)
            y.append(ys)
    else:
        for i in range(len(data)):
            end_ix = i + n_lags
            out_end_ix = end_ix + n_out
            # check if we are beyond the sequence
            if out_end_ix > len(data):
                break
            xs, ys = data[i:end_ix], data[end_ix:out_end_ix]
            X.append(xs)
            y.append(ys)

    return np.array(X), np.array(y)


def buildLSTModel(type_, n_neurons, dropout, n_lags, n_features, n_out):
    model = None
    if type_ == 'SimpleStateless':
        model = Sequential()
        model.add(LSTM(n_neurons, activation='relu', input_shape=(n_lags, n_features), recurrent_dropout=dropout,
                       stateful=False))
        model.add(Dense(1))
    if type_ == 'StackedStateless':
        model = Sequential()
        model.add(LSTM(n_neurons, activation='relu', return_sequences=True, input_shape=(n_lags, n_features),
                       recurrent_dropout=dropout, stateful=False))
        model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, stateful=False))
        model.add(Dense(1))
    if type_ == 'Bidirectional':
        model = Sequential()
        model.add(Bidirectional(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, stateful=False),
                                input_shape=(n_lags, n_features)))
        model.add(Dense(1))
    if type_ == 'Vector':
        model = Sequential()
        model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, return_sequences=True,
                       input_shape=(n_lags, n_features)))
        model.add(LSTM(n_neurons, recurrent_dropout=dropout, activation='relu'))
        model.add(Dense(n_out))
    if type_ == 'Encoder-Decoder':
        model = Sequential()
        model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, input_shape=(n_lags, n_features)))
        model.add(RepeatVector(n_out))
        model.add(LSTM(n_neurons, activation='relu', recurrent_dropout=dropout, return_sequences=True))
        model.add(TimeDistributed(Dense(1)))
    if model is not None:
        model.compile(optimizer='adam', loss='mse')
    return model


# # scale train and test data to [-1, 1]
# def scale(train, test):
# 	# fit scaler
# 	scaler = MinMaxScaler(feature_range=(-1, 1))
# 	scaler = scaler.fit(train)
# 	# transform train
# 	train = train.reshape(train.shape[0], train.shape[1])
# 	train_scaled = scaler.transform(train)
# 	# transform test
# 	test = test.reshape(test.shape[0], test.shape[1])
# 	test_scaled = scaler.transform(test)
# 	return scaler, train_scaled, test_scaled
#
# # inverse scaling for a forecasted value
# def invert_scale(scaler, X, yhat):
# 	new_row = [x for x in X] + [yhat]
# 	array = numpy.array(new_row)
# 	array = array.reshape(1, len(array))
# 	inverted = scaler.inverse_transform(array)
# 	return inverted[0, -1]

def experiment(type_, df_, lg, shf, nout):
    drop = [0.1, 0.2, 0.3]
    neurons = [100, 500, 1000]
    reps = 30
    epchs = [100, 500, 1000]
    # define input sequence
    raw_seq = df_.iloc[range(len(df_.index)), 1]
    raw_seq = raw_seq.to_numpy()
    train = raw_seq[0:-14]
    test = raw_seq[-14:]
    features = 1
    minerr = 1000000
    n_min = None
    d_min = None
    e_min = None
    l_min = None
    if nout is None:
        X_train, y_train = data_to_supervised(train, lg, None)
        X_test, y_test = data_to_supervised(test, lg, None)
    else:
        X_train, y_train = data_to_supervised(train, lg, nout)
        X_test, y_test = data_to_supervised(test, lg, nout)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], features))
    for n in neurons:
        # print("Neurons:", n)
        for d in drop:
            # print("drop:", d)
            for e in epchs:
                # print("Epochs:", e)
                error_scores = list()
                avg_predictions = np.full(shape=(len(y_test), 1), fill_value=0.0)
                for r in range(reps):
                    model = buildLSTModel(type_, n_neurons=n, dropout=d, n_lags=lg, n_features=features, n_out=nout)
                    model.fit(X_train, y_train, epochs=e, verbose=0, shuffle=shf)
                    yhat = model.predict(X_test, verbose=0)
                    if type_ == 'Encoder-Decoder':
                        yhat = yhat.reshape(yhat.shape[0], yhat.shape[1])
                    rmse = sqrt(mean_squared_error(yhat, y_test))
                    avg_predictions = avg_predictions + np.array(yhat)
                    # print('%d) Test RMSE: %.3f' % (r + 1, rmse))
                    error_scores.append(rmse)
                avg_predictions = avg_predictions / reps
                df = pd.DataFrame(avg_predictions)
                df.to_csv(
                    'predictions_' + type_ + '_' + str(lg) + '_lags_' + str(d) + "_drop_" + str(n) + "_neurons_" + str(
                        e) + "_epochs_" + str(shf) + "_shf" + '.csv', index=False, header=False)
                err = pd.DataFrame(error_scores)
                err.to_csv('rmse_' + type_ + "_" + str(lg) + '_lags_' + str(d) + "_drop_" + str(n) + "_neurons_" + str(
                    e) + "_epochs_" + str(shf) + "_shf" + '.csv', index=False, header=False)
                print('Avg. Test RMSE: %.3f l = %d n = %d d = %f e = %d  shf = %d' % (st.mean(error_scores), lg, n, d, e, shf))
                mean_err = st.mean(error_scores)
                if mean_err < minerr:
                    minerr = mean_err
                    n_min = n
                    d_min = d
                    e_min = e
                    l_min = lg

    print('Minimum Test RMSE: %.3f %d %d %f %d' % (minerr, l_min, n_min, d_min, e_min))


if __name__ == '__main__':
    dat = pd.read_excel('/Users/bogdanoancea/OneDrive/papers/2024/Olomouc/GDP_Q2024.xlsx')
    print(dat)
    lags=[1,2,3,4,5,6,7,8,9]
    # psimple = [0] * 9
    # i = 0
    pstacked = [0] * 9
    j = 0
    # pbidir = [0] * 9
    # k = 0
    #pvector = [0] * 6
    #lags = [4, 5, 6, 7, 8, 9]
    #ii = 0
    #pencoder = [0] * 6
    #lags = [6]
    #jj = 0
    for l in lags:
        # psimple[i] = Process(target=experiment, args = ('Bidirectional', dat, l, False, None))
        # psimple[i].start()
        # i = i +1
        pstacked[j] = Process(target=experiment, args = ('StackedStateless', dat, l, False, None))
        pstacked[j].start()
        j = j + 1
        # pbidir[k] = Process(target=experiment, args = ('Bidirectional', dat, l, False, None))
        # pbidir[k].start()
        # k = k + 1

        # pvector[ii] = Process(target=experiment, args=('Vector', dat, l, True, 3))
        # pvector[ii].start()
        # ii = ii + 1

        #pencoder[jj] = Process(target=experiment, args=('Encoder-Decoder', dat, l, True, 3))
        #pencoder[jj].start()
        #jj = jj + 1

    # for p in psimple:
    #     p.join()
    for p in pstacked:
         p.join()
    # for p in pbidir:
    #     p.join()
    # for p in pvector:
    #     p.join()
    #for p in pencoder:
    #    p.join()

# print('Plotting Results')
# plt.subplot(2, 1, 1)
# plt.plot(expected_output)
# plt.title('Expected')
# plt.subplot(2, 1, 2)
# plt.plot(predicted_output)
# plt.title('Predicted')
# plt.show()


from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

# Function to create the Keras model for SciKeras
def create_model(optimizer='adam', lstm_neurons=50, activation='relu'):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(lstm_neurons, activation=activation, input_shape=(seq_len, features)))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Keras model with SciKeras wrapper
model = KerasRegressor(model=create_model, epochs=25, batch_size=1, verbose=2)

# Hyperparameters to be optimized
param_grid = {
    #'model__optimizer': ['adam'],      # Note the prefix "model__"
    'model__lstm_neurons': [10, 25 , 50 , 100],         # Note the prefix "model__"
    'model__activation': ['relu', 'sigmoid'],        # Note the prefix "model__"
    'batch_size': [1],
    'epochs': [10]
}

# GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=2)
grid_result = grid.fit(X_train, y_train)
# Display the best hyperparameters
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Extract results from GridSearchCV
results = pd.DataFrame(grid_result.cv_results_)