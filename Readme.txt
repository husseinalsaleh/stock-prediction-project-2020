In order to run the follwoing project inside the IDE please make sure to download the follwoing:

pip install numpy
pip install scipy
pip install tensorflow
pip install pandas
pip install keras
pip install matplotlib
pip install sklearn
pip install PyQt5
المكتبات التي نحتاجها لتشغيل الكود

import tensorflow
import sys
import preprocess_data as ppd
from datetime import datetime
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from pandas_datareader import data, wb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
import seaborn as sns
