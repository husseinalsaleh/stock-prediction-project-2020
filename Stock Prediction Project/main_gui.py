import sys
import preprocess_data as ppd
from Stock_Price_Predictor import stocks
from gui import Ui_main
ui = Ui_main
from datetime import datetime
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from gui import Ui_main
import pandas_datareader as web
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
###################################
###################################
class stock_predacte(QtWidgets.QMainWindow, Ui_main):
    predictionModel = []
    data = []
    model = []

    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.handel_buttons()
        self.handel_ui_changes()

    def handel_ui_changes(self):
        self.tabWidget.tabBar().setVisible(False)

    def handel_buttons(self):
        self.pushButton.clicked.connect(self.open_home_tab)
        self.pushButton_18.clicked.connect(self.open_train_tab)
        self.pushButton_4.clicked.connect(self.open_train_tab)
        self.pushButton_2.clicked.connect(self.open_prediction_tab)
        self.pushButton_3.clicked.connect(self.open_graph_tab)
        self.pushButton_20.clicked.connect(self.chose_company)
        self.pushButton_17.clicked.connect(self.train_model)
        self.pushButton_16.clicked.connect(self.train_improved_LSTM_model)
        self.pushButton_6.clicked.connect(self.data_prediction)
        self.pushButton_19.clicked.connect(self.next_day_Prediction)
        self.pushButton_21.clicked.connect(self.close_Prediction)
        self.pushButton_12.clicked.connect(self.graph0)
        self.pushButton_13.clicked.connect(self.graph1)
        self.pushButton_14.clicked.connect(self.graph2)
        self.pushButton_15.clicked.connect(self.graph3)
        self.pushButton_8.clicked.connect(self.graph4)
        self.pushButton_5.clicked.connect(self.exit)
        #########################################

        ######################################

    def open_home_tab(self):
        self.tabWidget.setCurrentIndex(0)

    def open_train_tab(self):
        self.tabWidget.setCurrentIndex(1)

    def open_prediction_tab(self):
        self.tabWidget.setCurrentIndex(2)

    def open_graph_tab(self):
        self.tabWidget.setCurrentIndex(3)

    #######################################

    def chose_company(self):
        stockName = str(self.comboBox.currentText())
        data = web.DataReader(stockName, data_source='yahoo', start='2000-01-01',
                              end=datetime.date(datetime.now()))
        #     data.to_csv('data.csv', index=False)
        #     data = pd.read_csv('data.csv')
        stocks = ppd.remove_data(data)
        # Normalise the data using minmaxscaler function
        stocks = ppd.get_normalised_data(stocks)
        stocks.to_csv('data_preprocessed.csv', index=False)

    ########################################
    ########################################
    ########################################
    def train_model(self):
        from Stock_Price_Predictor import model, X_train, y_train
        model.fit(
            X_train,
            y_train,
            batch_size=1,
            epochs=1,
            validation_split=0.05)
        self.textEdit.setText(str("Training Completed"))

    ########################################
    ########################################
    def train_improved_LSTM_model(self):
        from Stock_Price_Predictor import model, X_train, y_train, batch_size, epochs

        model.fit(X_train,
                  y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  validation_split=0.05
                  )
        self.textEdit_7.setText(str("Training Completed"))
        # self.textEdit_8.setText(str("you can get the result"))

    #########################################
    #########################################
    def data_prediction(self):
        import preprocess_data as ppd
        import stock_data as sd
        import numpy as np
        import math
        import pandas as pd
        import Stock_Price_Predictor
        from Stock_Price_Predictor import model, X_train, y_train, X_test, y_test, stocks_data
        trainScore = model.evaluate(X_train, y_train, verbose=0)

        testScore = model.evaluate(X_test, y_test, verbose=0)

        range = [np.amin(stocks_data['Close']), np.amax(stocks_data['Close'])]

        # Calculate the stock price delta in $

        true_delta = testScore * (range[1] - range[0])
        data = pd.read_csv('data_preprocessed.csv')
        stocks = ppd.remove_data(data)
        stocks = ppd.get_normalised_data(stocks)

        stocks = stocks.drop(['Item'], axis=1)
        self.textEdit_2.setText(str(stocks.head(28)))

        # model = LinearRegressionModel.build_model(X_train, y_train)

        X = stocks[:].values
        Y = stocks[:]['Close'].values

        X = sd.unroll(X, 1)
        Y = Y[-X.shape[0]:]

        # Generate predictions
        predictions = Stock_Price_Predictor.model.predict(X)

        # get the test score
        testScore = Stock_Price_Predictor.model.evaluate(X, Y, verbose=0)

        self.textEdit_3.setText(str(testScore))
        self.textEdit_4.setText(str(math.sqrt(testScore)))
        self.textEdit_5.setText(str(trainScore))
        self.textEdit_6.setText(str(true_delta))

    ##########################################
    def close_Prediction(self):
        data = pd.read_csv('data_preprocessed.csv')
        stocks = ppd.remove_data(data)
        stocks = ppd.get_normalised_data(stocks)
        stocks = stocks[['Close']]
        future_days = 25
        stocks['Prediction'] = stocks[['Close']].shift(-future_days)
        self.textEdit_10.setText(str(stocks.head(28)))

    ##########################################
    def next_day_Prediction(self):
        # Create a new dataframe with only the 'Close' column
        data = stocks.filter(['Close'])
        # Converting the dataframe to a numpy array
        dataset = data.values
        # Get /Compute the number of rows to train the model on
        training_data_len = math.ceil(len(dataset) * .8)
        # Scale the all of the data to be values between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        train_data = scaled_data[0:training_data_len, :]
        x_train = []
        y_train = []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60:i, 0])
            y_train.append(train_data[i, 0])
        model = Sequential()

        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        # Create a new dataframe
        new_df = stocks.filter(['Close'])
        # Get teh last 60 day closing price
        last_60_days = new_df[-60:].values
        # Create an empty list
        X_test = []
        # Append teh past 60 days
        X_test.append(last_60_days)
        # Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        # Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        # Get the predicted scaled price
        pred_price = model.predict(X_test)
        self.textEdit_9.setText(str(pred_price))

    ##########################################
    def graph0(self):
        import visualize
        from Stock_Price_Predictor import stocks
        visualize.plot_basic(stocks)

    ##########################################
    def graph1(self):
        import preprocess_data as ppd
        import visualize
        from Stock_Price_Predictor import stocks
        stocks = ppd.get_normalised_data(stocks)
        visualize.plot_basic(stocks)

    ###########################################
    def graph2(self):
        import pandas as pd
        import LinearRegressionModel
        import stock_data as sd
        import visualize
        stocks = pd.read_csv('data_preprocessed.csv')
        X_train, X_test, y_train, y_test, label_range = sd.train_test_split_linear_regression(stocks)
        model = LinearRegressionModel.build_model(X_train, y_train)
        predictions = LinearRegressionModel.predict_prices(model, X_test, label_range)
        visualize.plot_prediction(y_test, predictions)

    ###########################################
    def graph3(self):
        import visualize
        from Stock_Price_Predictor import model, X_test, y_test
        predictions = model.predict(X_test)
        visualize.plot_lstm_prediction(predictions, y_test)

    ###########################################
    def graph4(self):
        import visualize as vs
        from Stock_Price_Predictor import model, X_test, y_test
        batch_size = 512

        predictions = model.predict(X_test, batch_size=batch_size)

        vs.plot_lstm_prediction(predictions, y_test)

    #########################################

    #########################################
    def exit(self):
        qApp.closeAllWindows()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = stock_predacte()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
