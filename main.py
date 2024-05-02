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
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


df = pd.read_excel('/Users/bogdanoancea/OneDrive/papers/2024/time-series/date.xlsx')

print(df)

ts = df.iloc[:, [1, 3, 4]]
ts.rename(columns={'rata inflatiei': 'Inflation', 'indice sentiment': 'Sentiment','rata somajului': 'Unemployment'}, inplace=True)

columns_titles = ["Sentiment","Unemployment", "Inflation"]
ts=ts.reindex(columns=columns_titles)

# choose a number of time steps
n_steps_in, n_steps_out = 6, 4
# covert into input/output
X, y = split_sequences(ts.to_numpy(), n_steps_in, n_steps_out)
print(X.shape, y.shape)
# summarize the data
for i in range(len(X)):
	print(X[i], y[i])

n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(1500, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(1500, activation='relu', recurrent_dropout=0.15))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse',  metrics=['mae', 'mape', 'mse'])
# fit model
history = model.fit(X[:-1,:], y[:-1,:], epochs=250, verbose=1, shuffle = False)
# demonstrate prediction
x_input = X[62]
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

plt.plot(history.history['mse'])
plt.plot(history.history['mae'])
plt.plot(history.history['mape'])
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



hist_df = pd.DataFrame(history.history)

# save to json:
hist_json_file = 'history.json'
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

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

ypredtr = model2.predict(X[:-1,:])
plt.plot(ypredtr)
plt.plot(y[:-1,:])
plt.show()

test_mse = sklearn.metrics.mean_squared_error(y[62], yhat2[0])

plt.plot(yhat2[0])
plt.plot(y[62])
plt.show()







from sklearn.ensemble import RandomForestRegressor
modelrf= RandomForestRegressor(n_estimators=100)

modelSVR = SVR(kernel='poly',degree=8, C=1)
xtrain = ts.iloc[0:68, 0:2]
ytrain = ts.iloc[0:68, 2:3]
modelSVR.fit(xtrain, ytrain)
modelrf.fit(xtrain, ytrain)
# demonstrate prediction
x_test =ts.iloc[68:72, 0:2]
y_test = ts.iloc[68:72, 2:3]
yhat2 = modelrf.predict(x_test)

print(yhat2)

regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(X, y)