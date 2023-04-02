import sys
import tensorflow as tf
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from flask_caching import Cache

## for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
app = Flask(__name__)
CORS(app)

# Configure the cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/predict2', methods=['GET'])
@cache.cached(timeout=300)  # Cache the response for 5 minutes
def predict2():
    df = pd.read_excel("22yearsdata.xlsx")
    col = df.columns
    print(df.describe())
    df = df.dropna()
    df = df.iloc[:, 1:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df)
    df = pd.DataFrame(df, columns=col[1:])
    df.head


    Predictors = df.iloc[:, :-1]
    print(Predictors)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(Predictors)

    Predictions = df.iloc[:, -1]
    print(Predictions)

    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    df_pca['target'] = Predictions



    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        dff = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(dff.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(dff.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    a = pd.concat([Predictors, Predictions], axis=1)
    reframed = series_to_supervised(a, 1, 1)
    # drop columns we don't want to predict
    # reframed.drop(reframed.columns[[11,12,13,14,15,16,17,18,19,20]], axis=1, inplace=True)
    print(reframed.head)

    # split into train and test sets
    values = reframed.values

    n_train_time = 300
    train = values[:n_train_time, :]
    test = values[n_train_time:, :]
    ##test = values[n_train_time:n_test_time, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


    # Load the LSTM TensorFlow model
    model = tf.keras.models.load_model('model')

    # make a prediction
    yhat = model.predict(test_X)

    test_X = test_X.reshape((test_X.shape[0], 21))
    # invert scaling for forecast
    inv_yhat = np.concatenate((test_X[:, -10:], yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, -1]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_X[:, -10:], test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, -1]

    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    my_df = pd.DataFrame(inv_y)
    my_df.describe

    interval = 5508
    aa = [x for x in range(interval)]

    def recursive_forecast(model, input_data, n_days):
        forecast = []
        current_input = input_data[-1].copy()

        for _ in range(n_days):
            prediction = model.predict(current_input.reshape(1, 1, -1))
            forecast.append(prediction[0, 0])
            current_input = np.roll(current_input, -1)
            current_input[-1] = prediction

        return np.array(forecast)

    n_days = 6
    yhat = recursive_forecast(model, test_X, n_days)
    # Reshape yhat to (1, n_days)
    yhat = yhat.reshape(-1, 1)

    inv_yhat = np.concatenate((test_X[:n_days, -10:], yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, -1]

    print(inv_yhat)
    interval = 100
    predictinterval = interval + n_days
    aa = [x for x in range(interval)]
    bb = [x for x in range(interval - 1, predictinterval)]
    print(bb)

    return jsonify({"next6days": inv_yhat.tolist()})

if __name__ == '__main__':
  app.run(debug=True)
