# Part 1 - Data Preprocessing
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# creating a data structure with 60 time steps and 1 output (60 timesteps corresponds to 3 months)
x_train = []
y_train = []
for i in range(60, 1258):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Part 2 - Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN
regressor = Sequential()

# Adding the first LSTM layer and adding some Dropout Regularization
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2)) # Dropout 20% of the neurons of the LSTM layer.


# Adding the second LSTM layer and adding some Dropout Regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) # Dropout 20% of the neurons of the LSTM layer.


# Adding the third LSTM layer and adding some Dropout Regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) # Dropout 20% of the neurons of the LSTM layer.


# Adding the fourth LSTM layer and adding some Dropout Regularization
regressor.add(LSTM(units = 50)) # no more return_sequences required, since this is the last layer of the RNN.
regressor.add(Dropout(0.2)) # Dropout 20% of the neurons of the LSTM layer.

# Adding the output layer
regressor.add(Dense(units = 1)) # the number of neurons needed for the output layer; i.e one; stock price at time t+1


# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the training set
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)


# Part 3 - Making the predictions and visualizing the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values


# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

# Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make the predictions based on X_test
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Vizualising the results

## Plot the Real Stock Price
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')

## Plot the Predicted Stock Price
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')

## Applying the Root Mean Squared Error
# import math
# from sklearn.metrics import mean_squared_error
# rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
# plt.plot(rmse, color = 'green', label = 'Root Mean Squared Value')

## Add a title to the Chart.
plt.title('Google Stock Price Prediction')

## Add x & y labels to the Chart.
plt.xlabel('Time')
plt.ylabel('Google Stock Price')

## Show the legend.
plt.legend

## Show the Chart.
plt.show