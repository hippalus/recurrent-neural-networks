import pandas as pd
from pandas import DataFrame
from pandas import concat
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import math
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

dataSet = pd.read_csv('../data/data.csv')
test = DataFrame(dataSet['Value'])


def data_series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


data = data_series_to_supervised(test, 16)
print(data)
inputs = data.iloc[:, [0, 1, 2]].values
outputs = data.iloc[:, [3]].values

trainSize = int(len(data) * 0.7)
print("data length", len(data))
print("input length", len(inputs))
print("output length", len(outputs))
test_size = len(data) - trainSize

trainX, trainY, testX, testY = inputs[0:trainSize, :], outputs[0:trainSize, :], inputs[trainSize:len(data), :], outputs[
                                                                                                                trainSize:len(
                                                                                                                    data),
                                                                                                                :]
print(trainX)
X = trainX
y = trainY
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=3))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
es = EarlyStopping(monitor='loss', mode='min', verbose=1)
model.fit(X, y, epochs=2000, verbose=0, callbacks=[es])
# demonstrate prediction
XInput = array([50, 60, 70])
XInput = XInput.reshape((1, 3))

# trainPredictMlp
trainPredictMlp = model.predict(X, verbose=0)
# testPredict
testPredictMlp = model.predict(testX, verbose=0)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y[:, 0], trainPredictMlp[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredictMlp[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))
timeStep = 16
trainPredictPlotMlp = np.empty_like(test)
trainPredictPlotMlp[:, :] = np.nan
trainPredictPlotMlp[timeStep:len(trainPredictMlp) + timeStep, :] = trainPredictMlp
# shift test predictions for plotting
testPredictPlotMlp = np.empty_like(test)
testPredictPlotMlp[:, :] = np.nan

testPredictPlotMlp[len(trainPredictMlp) + timeStep - 1:len(test) - 1, :] = testPredictMlp
# plot baseline and predictions
plt.plot(test)
plt.plot(trainPredictPlotMlp, color="yellow")
plt.plot(testPredictPlotMlp, color="red")

plt.show()

print(testPredictMlp)
print(testY)
