from scipy.stats import kde
from statsmodels import robust
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
from scipy.optimize import least_squares
from scipy.stats import linregress
from scipy.stats import skewnorm
import scipy.stats as stats


def norm(x):
  #array = np.array(train_stats['std'])
  #for i in x:
    #if array[i] == 0.:
      #return 0.
    #else:
  stats = np.loadtxt('../ChiDists.dat')
  mean = stats[:, 0]
  deviation = stats[:, 1]
  return (x - mean) / deviation



def ApplyModel(Agemodel, Zmodel, data, gals):

  #import the trained models, the appliction set and the galaxies in the application set
  stats = np.loadtxt('../ChiDists.dat')
  mean = stats[:, 0]
  deviation = stats[:, 1]
  #File to get r_eff
  rnames = ['Galaxy', 'arcsec', 'Reff']
  Reff=pd.read_table('../reff.dat', delim_whitespace = True, names = rnames)

  #empties for returning
  dztrue = []
  dzpred = []
  datrue = []
  dapred = []
  MCzgrad = []
  MCagrad = []
  lowagrad = []
  lowzgrad = []
  highagrad = []
  highzgrad = []
  zerror = []
  ageerror = []
  RZ = []
  RA = []
  name = []

  #compile saved NNs
  Zmodel.compile(loss='mse',
                  optimizer=tf.train.RMSPropOptimizer(0.001),
                  metrics=['mae', 'mse'])
  Agemodel.compile(loss='mse',
                  optimizer=tf.train.RMSPropOptimizer(0.001),
                  metrics=['mae', 'mse'])

  #Iterate over each galaxy
  for phi in gals:
    #print(phi)

    galaxy = data.loc[data['name'].isin([phi])]#Select galaxies needed
    galaxy = galaxy.drop(columns = 'name')
  
    galaxy = norm(galaxy)#normalisation
    #print('norm worked')
    thisr = Reff.loc[Reff['Galaxy'].isin([phi])]
    reff=  thisr['Reff'].values #r_eff for this galaxy

    smaller = galaxy.sample(frac = 1, random_state = 0) #Randomise order of dataset
 
  
    #Running the model on Z *****************************************
    zdataset = smaller.copy()
    zdataset = zdataset.dropna()
    zdataset = zdataset.drop(columns = 'age')
  
    ztrain_set2 = zdataset.sample(frac = 0.5, random_state = 0) #Split in two for application
    ztest_set2 = zdataset.drop(ztrain_set2.index)
  
    ztrain_positions = ztrain_set2.copy() #Save copies for spaxel position info
    ztest_positions = ztest_set2.copy()

    ztrain_set2 = ztrain_set2.drop(columns = 'xkpc')
    ztrain_set2 = ztrain_set2.drop(columns = 'ykpc')
    ztest_set2 = ztest_set2.drop(columns = 'xkpc')
    ztest_set2 = ztest_set2.drop(columns = 'ykpc')
  

    #train_stats2 = ztrain_set2.describe()
    #train_stats2.pop('ZHL')
    #train_stats2 = train_stats2.transpose()

    ztestlabel2 = ztest_set2.pop('ZHL') #Output labels
    ztrainlabel2 = ztrain_set2.pop('ZHL')

    znormtest = ztest_set2
    znormtrain = ztrain_set2
    #print(znormtest)
  
    zpredtest = Zmodel.predict(znormtest) #Apply the model
    zpredtrain = Zmodel.predict(znormtrain)
    #print(zpredtrain)

    zpred = np.concatenate((zpredtrain, zpredtest), axis = 0)#Make outputs back into one array
    zpred = zpred.flatten() #predicted Z values


    Z = np.concatenate((ztrainlabel2, ztestlabel2), axis = 0).flatten()#flux[:, 1] #True values for Z

  
    zpred = zpred * deviation[1] + mean[1] #Undo normalisation
    Z = Z * deviation[1] + mean[1]

    ztrain_pos = ztrain_positions.values#Position arrays
    x_train = ztrain_pos[:, -2]
    y_train = ztrain_pos[:, -1]  
    ztest_pos = ztest_positions.values#Position arrays
    x_test = ztest_pos[:, -2]
    y_test = ztest_pos[:, -1]
  
    xz = np.concatenate((x_train, x_test), axis = 0).flatten() #Positions into one array
    yz = np.concatenate((y_train, y_test), axis = 0).flatten()
  
    rz = np.sqrt(xz*xz + yz*yz)/reff[0] #Convert 2D position  into r
  
    for i in rz:
      RZ.append(i) #Position array for Z
  
  
    #Running the model on age ********************************************************************
    adataset = smaller.copy()
    adataset = adataset.dropna()
    adataset = adataset.drop(columns = 'ZHL')
  
    atrain_set2 = adataset.sample(frac = 0.5, random_state = 0) #Split array into 2 randomly
    atest_set2 = adataset.drop(atrain_set2.index)
  
    atrain_positions = atrain_set2.copy() #Save arrray positions in order
    atest_positions = atest_set2.copy()

    atrain_set2 = atrain_set2.drop(columns = 'xkpc')
    atrain_set2 = atrain_set2.drop(columns = 'ykpc')
    atest_set2 = atest_set2.drop(columns = 'xkpc')
    atest_set2 = atest_set2.drop(columns = 'ykpc')
  

    #train_stats2 = atrain_set2.describe()
    #train_stats2.pop('age')
    #train_stats2 = train_stats2.transpose()

    atestlabel2 = atest_set2.pop('age') #Labels
    atrainlabel2 = atrain_set2.pop('age')

    anormtest = atest_set2#norm(atest_set2)
    anormtrain = atrain_set2#norm(atrain_set2)
  
    apredtest = Agemodel.predict(anormtest) #Predictions made
    apredtrain = Agemodel.predict(anormtrain)


    apred = np.concatenate((apredtrain, apredtest), axis = 0)
    apred = apred.flatten()#Predicted age in 1 array


    age = np.concatenate((atrainlabel2, atestlabel2), axis = 0).flatten()#flux[:, 1] #True values for age
    apred = apred*deviation[0] + mean[0]#Un-normalising
    age = age * deviation[0] + mean[0]
        
      
    atrain_pos = atrain_positions.values#Position arrays
    x_train = atrain_pos[:, -2]
    y_train = atrain_pos[:, -1]  
    atest_pos = atest_positions.values#Position arrays
    x_test = atest_pos[:, -2]
    y_test = atest_pos[:, -1]
  
    xa = np.concatenate((x_train, x_test), axis = 0).flatten()
    ya = np.concatenate((y_train, y_test), axis = 0).flatten()
  
    ra = np.sqrt(xa*xa + ya*ya)/reff[0] #radial position in 1 array
  
    for i in ra:
      RA.append(i)
  
  
    #************************************************************
   
    #Values for outputting for summaries
    for i in range(0, len(zpred)):
      dzpred.append(zpred[i])
      dztrue.append(Z[i])
      dapred.append(apred[i])
      datrue.append(age[i])
      name.append(phi)
  
  #print(len(name), len(dzpred), len(dapred), len(datrue)) 
  output = pd.DataFrame({'name': name, 'dapred': dapred, 'datrue': datrue, 'dzpred': dzpred, 
			 'dztrue': dztrue, 'RA': RA, 'RZ': RZ})
  
  return output
 
