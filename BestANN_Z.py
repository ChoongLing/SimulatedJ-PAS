
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
#import matplotlib
#matplotlib.use('agg')
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

def build_model(): #Form of the ANN
  model = keras.Sequential([
    layers.Dense(35, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dropout(0.05),
    layers.Dense(35, activation=tf.nn.relu), #second hidden layer
    layers.Dropout(0.05),
    layers.Dense(35, activation=tf.nn.relu), #third hidden layer
    layers.Dropout(0.05),
    layers.Dense(35, activation=tf.nn.relu), #fourth hidden layer
    layers.Dropout(0.05),
    layers.Dense(35, activation=tf.nn.relu), #fifth hidden layer
    layers.Dropout(0.05),
    layers.Dense(1) #output layer
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  
  return model


def norm(x):
  return (x - mean) / deviation

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
incoming = pd.read_table('../Alldata.dat', delim_whitespace = True, names = column_names)
incoming = incoming.drop(columns = 'log age (L)')
#Masking H_alpha
incoming = incoming.drop(columns = 'JPAS6600_145')
incoming = incoming.drop(columns = 'JPAS6700_145')
dataset = incoming.copy()
dataset = dataset.dropna()
ForUse = dataset.copy()

#Galaxies to be used in the training sample
Jgals = open('../GroupJ/UnbrokenJ.txt', 'r')
Kgals = open('../GroupK/UnbrokenK.txt', 'r')
Igals = open('../GroupI/UnbrokenI.txt', 'r')
notL = []
for i in Jgals:
  split = i.split()
  notL.append(split[0])
for i in Kgals:
  split = i.split()
  notL.append(split[0])
for i in Igals:
  split = i.split()
  notL.append(split[0])
  
datasetA = dataset.copy()

GroupA = datasetA.loc[dataset['name'].isin(notL)]# Dataset for trainig upon
GroupA = GroupA.drop(columns = 'name')

smalldataset1 = GroupA.sample(frac = 1, random_state = 0) #Reduce the size of training set?

smalldataset = smalldataset1[smalldataset1.ZHL > -1.]#metallicity cut
smalldataset = smalldataset.drop(columns = 'xkpc')
smalldataset = smalldataset.drop(columns = 'ykpc')
#split dataset into training & testing
train_dataset = smalldataset.sample(frac=0.8,random_state=0)
test_dataset = smalldataset.drop(train_dataset.index)



#Gives the statistics of each band, not particularly relevant now
train_stats = train_dataset.describe()
train_stats.pop("ZHL")
train_stats = train_stats.transpose()

#True values for Z
train_labels = train_dataset.pop("ZHL")
test_labels = test_dataset.pop("ZHL")

#Normalise dataset
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

#Building the model & defining parameters
model = build_model()
earlystop = EarlyStopping(monitor='mean_absolute_error', patience=200)
callbacks_list = [earlystop]
EPOCHS =5000 #Number of iterations

#fitting the model
history = model.fit(
  normed_train_data, train_labels, batch_size = 64, callbacks = callbacks_list,
  epochs=EPOCHS, validation_split = 0.2, verbose=0)

#tracks the values of errors, variable values etc.
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

model.save('ZForGroupLUnbroken.h5') #Save the model for use in trsting

#Error plot
top = figure(plot_width=400, plot_height = 400, title = 'Error')
top.line(hist['epoch'], hist['mean_absolute_error'], color = 'blue')
top.line(hist['epoch'], hist['val_mean_absolute_error'], color = 'orange')
top.xaxis.axis_label = 'Epoch'
top.yaxis.axis_label = 'Mean abs error'

test_predictions = model.predict(normed_test_data).flatten() #Make predictions on validation set
#Fit to the predictions
fit = np.polyfit(test_labels, test_predictions, 1)
slope = fit[0]
intercept = fit[1]
stddev = np.std(test_predictions, dtype = np.float16)
xfit = [min(test_labels), max(test_labels)]
yfit = [slope * xx + intercept for xx in xfit]  
error = test_predictions - test_labels
stddev = np.std(error)

#True vs Pred plot
mid = figure(plot_width=400, plot_height = 400, title = str(slope) + ', ' + str(stddev))
mid.cross(test_labels, test_predictions)
mid.line([-0.75, 0.2], [-0.75, 0.2], color = 'black')
mid.line(xfit, yfit, color = 'blue')
mid.xaxis.axis_label = 'True Value'
mid.yaxis.axis_label = 'Predicted Value'

p = gridplot([[top, mid]])

bokeh.io.save(p, filename = 'ModelZForLUnbroken.html')

#Same plots but giving .png instead of .html
plt.plot(hist['epoch'], hist['mean_absolute_error'], color = 'black')
plt.plot(hist['epoch'], hist['val_mean_absolute_error'], color = 'red')
plt.xlabel('Epoch')
plt.ylabel('Mean abs error')
plt.savefig('Training/ZTrainedErrorLUnbroken.png')

plt.clf()
plt.scatter(test_labels, test_predictions, color = 'grey')
plt.plot([-1, 0.2], [-1, 0.2], color = 'black')
plt.plot(xfit, yfit, color = 'red')
plt.xlabel('True Age')
plt.ylabel('Predicted Age')
plt.savefig('Training/ZTrainedTrueVsLongerLUnbroken.png')

plt.clf()
#Plot a gaussian to show errors
error = test_predictions - test_labels
mu = np.mean(error)
sigma = np.std(error)
result = plt.hist(error, bins = 25, facecolor = 'black')
xa = np.linspace(min(error), max(error), 200)
dx = result[1][1] - result[1][0]
scale = len(error) * dx
plt.plot(xa, mlab.normpdf(xa, mu, sigma) * scale, color = 'red')
#plt.text(-1.0, 4000, "Std dev = " +str(stddev))
plt.xlabel("Prediction Error")
plt.ylabel("Count")
#plt.title('log age (L)')
plt.savefig('Training/ZTrainedHistLUnbroken.png')
