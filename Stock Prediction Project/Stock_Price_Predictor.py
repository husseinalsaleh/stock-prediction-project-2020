# Preprocess the data
# %%
# Bench Mark Model
# %%

# # Normalise the data using minmaxscaler function
import pandas as pd
from sklearn.metrics import mean_squared_error

import LinearRegressionModel
import stock_data as sd

#stocks = pd.read_csv('data_preprocessed.csv')
linearstocks = pd.read_csv('data_preprocessed.csv',nrows=1250)

## split the data
X_train, X_test, y_train, y_test, label_range = sd.train_test_split_linear_regression(linearstocks)


# train a linear regresson

model = LinearRegressionModel.build_model(X_train, y_train)

# get predict on test set

predictions = LinearRegressionModel.predict_prices(model, X_test, label_range)

# measure accuracy of the prediction

trainScore = mean_squared_error(X_train, y_train)

testScore = mean_squared_error(predictions, y_test)


## Long-Sort Term Memory Model
# In this section we will use LSTM to train and test on our data set.

## Basic LSTM Model
### First lets make a basic LSTM model.

# import keras libraries for smooth implementaion of lstm
# %%
import pandas as pd

import lstm, time  # helper libraries

import stock_data as sd

stocks = pd.read_csv('data_preprocessed.csv')
stocks_data = stocks.drop(['Item'], axis=1)



#Split train and test data sets and Unroll train and test data for lstm model

# %%
X_train, X_test, y_train, y_test = sd.train_test_split_lstm(stocks_data, 5)

unroll_length = 50
X_train = sd.unroll(X_train, unroll_length)
X_test = sd.unroll(X_test, unroll_length)
y_train = y_train[-X_train.shape[0]:]
y_test = y_test[-X_test.shape[0]:]



#Build a basic Long-Short Term Memory model

# build basic lstm model
model = lstm.build_basic_model(input_dim=X_train.shape[-1], output_dim=unroll_length, return_sequences=True)

# Compile the model
start = time.time()
model.compile(loss='mean_squared_error', optimizer='adam')
#make prediction using test data

predictions = model.predict(X_test)
# Get the test score.
# %%
trainScore = model.evaluate(X_train, y_train, verbose=0)

testScore = model.evaluate(X_test, y_test, verbose=0)
# %% md
## Improved LSTM Model
# Build an improved LSTM model
# Set up hyperparameters
batch_size = 512
epochs = 20
# build improved lstm model
model = lstm.build_improved_model(X_train.shape[-1], output_dim=unroll_length, return_sequences=True)
start = time.time()
model.compile(loss='mean_squared_error', optimizer='adam')
predictions = model.predict(X_test, batch_size=batch_size)

################################################
