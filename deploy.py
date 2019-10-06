import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


def test():
    print('aaa')
    dataset_train = pd.read_csv("bitcoin.csv")
    training_set = dataset_train.iloc[:,1:2].values

    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_training_set = scaler.fit_transform(training_set)

    # Creating new Data Structure
    X_train = []
    y_train = []

    for i in range(10,30):
        X_train.append(scaled_training_set[i-10:i, 0])
        y_train.append(scaled_training_set[i, 0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))

    # Building the Neural Network
    

    regressor = Sequential()
    regressor.add(LSTM(units = 50, return_sequences= True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences= True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences= True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))


    # Compiling and Fitting the RNN to the Training set
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.fit(X_train, y_train, epochs=100, batch_size=32)

    # Making Predictions
    # getting the Actual Stock Prices of Jan-2017
    dataset_test = pd.read_csv("bitcoin.csv")
    actual_stock_price = dataset_test.iloc[:,1:2].values

    # getting the Predicted Stock Prices of Jan-2017
        # Step1 - preparing the input for the model
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
    inputs = dataset_total[len(dataset_total)- len(dataset_test)-10:].values

    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(10,30):
        X_test.append(inputs[i-10:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1))

        # Step2 - prediction
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    # Visualising the Results

    plt.plot(actual_stock_price, color = 'red', label = 'Actual Google Stock Price')
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Google Stock Price')
    plt.legend()
    plt.show()

test()
