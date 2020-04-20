# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy
import math

dataSet = pandas.read_csv('../data/data.csv', usecols=[1], engine='python')
plt.plot(dataSet)
plt.show()

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataSet = dataSet.values.astype('float32')

# normalize the dataSet
scaler = MinMaxScaler(feature_range=(0, 1))
dataSet = scaler.fit_transform(dataSet)

"""The code below calculates the index of the split point and separates the data into the 
training datasets with 70% of the observations that we can use to train our model,
 leaving the remaining 30% for testing the model."""
# split into train and test sets
trainSize = int(len(dataSet) * 0.70)
testSize = len(dataSet) - trainSize
train, test = dataSet[0:trainSize, :], dataSet[trainSize:len(dataSet), :]
print('Test Size ' + str(len(test)), 'Train Size:' + str(len(train)))


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=5):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# reshape into X=t and Y=t+1
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
"""
The network has a visible layer with 1 input, a hidden layer with 4 LSTM blocks or neurons,
and an output layer that makes a single value prediction.
The default sigmoid activation function is used for the LSTM blocks.
The network is trained for 100 epochs and a batch size of 1 is used.
"""
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(6, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPrediction = model.predict(trainX)
testPrediction = model.predict(testX)

# invert predictions
trainPrediction = scaler.inverse_transform(trainPrediction)
trainY = scaler.inverse_transform([trainY])

testPrediction = scaler.inverse_transform(testPrediction)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPrediction[:, 0]))
print('Train Score: %.2f RMSE' % trainScore)
testScore = math.sqrt(mean_squared_error(testY[0], testPrediction[:, 0]))
print('Test Score: %.2f RMSE' % testScore)

"""
Finally, we can generate predictions using the model for both the train
and test dataset to get a visual indication of the skill of the model.

Because of how the dataset was prepared, we must shift the predictions 
so that they align on the x-axis with the original dataset. Once prepared,
the data is plotted, showing the original dataset in blue, the predictions for the training dataset in green,
and the predictions on the unseen test dataset in red.
"""

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataSet)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPrediction) + look_back, :] = trainPrediction

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataSet)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPrediction) + (look_back * 2) + 1:len(dataSet) - 1, :] = testPrediction

# plot baseline and predictions

plt.plot(scaler.inverse_transform(dataSet))
plt.plot(testPredictPlot)
plt.plot(trainPredictPlot)
plt.show()
