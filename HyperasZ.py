from hyperopt import Trials, STATUS_OK, tpe
import optim
from hyperopt.hp import choice
from hyperopt.hp import uniform
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns

import bokeh
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.models import Range1d


tf.enable_eager_execution()


def create_model(normed_train_data, normed_test_data, train_labels, test_labels):
  #Build the CNN
  model= keras.models.Sequential()
  model.add(layers.Reshape((32, 1))) #reshape for use in CNN
  # CNN layers
  model.add(layers.Conv1D(activation='relu',# kernel_initializer=initializer, 
		  padding="same", filters={{choice([8, 16, 20, 25, 30, 33, 35, 40, 45, 50, 55, 60, 64])}}, 
		  kernel_size={{choice([6, 8, 10, 12])}}))  
  model.add(layers.Conv1D(activation='relu',# kernel_initializer=initializer, 
		  padding="same", filters={{choice([8, 16, 20, 25, 30, 33, 35, 40, 45, 50, 55, 60, 64])}}, 
		  kernel_size={{choice([6, 8, 10, 12])}}))

  # Max pooling layer
  model.add(layers.MaxPooling1D(pool_size={{choice([2, 3,  4, 5,  6, 7, 8])}}))
  model.add(layers.Flatten())#flatten for use in dense layers
  
  # Dense layers
  model.add(layers.Dense(units={{choice([25, 30, 33, 35, 40])}},# kernel_initializer=initializer, 
		activation='relu'))
  model.add(layers.Dense(units=1, activation="linear"))#, #output layer

  optimizer = tf.train.RMSPropOptimizer(0.001)
  model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])  
  
  result = model.fit(normed_train_data, train_labels,
            batch_size={{choice([32, 64, 128])}},
            epochs=500,
            verbose=0,
            validation_split=0.2)

  validation_acc = np.amax(result.history['val_mean_absolute_error']) 
  print('Best validation acc of epoch:', validation_acc)
  return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

def data():
  #Load in the stats for normalisation
  stats = np.loadtxt('../Zstats.dat')
  mean = stats[:, 0]
  deviation = stats[:, 1]

  column_names = ['name', 'log age (L)', 'ZHL', 'JPAS4000_145', \
    'JPAS4100_145', 'JPAS4200_145', 'JPAS4300_145', 'JPAS4400_145', 'JPAS4500_145', 'JPAS4600_145', 'JPAS4700_145',\
    'JPAS4800_145', 'JPAS4900_145', 'JPAS5000_145', 'JPAS5100_145', 'JPAS5200_145', 'JPAS5300_145', 'JPAS5400_145',\
    'JPAS5500_145','JPAS5600_145', 'JPAS5700_145', 'JPAS5800_145', 'JPAS5900_145', 'JPAS6000_145', 'JPAS6100_145',\
    'JPAS6200_145', 'JPAS6300_145', 'JPAS6400_145', 'JPAS6500_145', 'JPAS6600_145', 'JPAS6700_145', 'JPAS6800_145', \
    'JPAS6900_145', 'JPAS7000_145', 'JPAS7100_145', 'JPAS7200_145', 'JPAS7300_145', 'xkpc', 'ykpc']

  feature_names = column_names[0] #column for metallicity
  label_names = column_names[1:] #columns for the filters

  #Import data
  incoming = pd.read_table('../Train1Data.dat', delim_whitespace = True, names = column_names)
  incoming = incoming.drop(columns = 'log age (L)')
  #Masking H_alpha
  incoming = incoming.drop(columns = 'JPAS6600_145')
  incoming = incoming.drop(columns = 'JPAS6700_145')
  dataset = incoming.copy()
  dataset = dataset.dropna()

  datasetA = dataset.copy()

  #Build dataset of galaxies
  GroupA = datasetA.copy()
  GroupA = GroupA.drop(columns = 'name')

  smalldataset1 = GroupA.sample(frac = 1, random_state = 0) #dataset of 8400 rows

  smalldataset = smalldataset1.drop(columns = 'xkpc')
  smalldataset = smalldataset.drop(columns = 'ykpc')
  
  #split dataset into training & testing
  train_dataset = smalldataset.sample(frac=0.8,random_state=0)
  test_dataset = smalldataset.drop(train_dataset.index)

  #Gives the statistics of each band, not particularly relevant now
  train_stats = train_dataset.describe()
  train_stats.pop("ZHL")
  train_stats = train_stats.transpose()

  train_labels = train_dataset.pop("ZHL")
  test_labels = test_dataset.pop("ZHL")

  #Normalise dataset
  normed_train_data = (train_dataset.values - mean )/ deviation
  normed_test_data = (test_dataset.values - mean)/deviation
  train_labels = train_labels.values
  test_labels = test_labels.values

  return normed_train_data, train_labels,  normed_test_data, test_labels


best_run, best_model = optim.minimize(model=create_model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=5,
                                      trials=Trials())
X_train, Y_train, X_test, Y_test = data()
print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, Y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)
