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




# Define fitting function (linear)
def lin_func(p, x):
     a, b = p
     return a*x + b

def resid(p, x, y):
    return p[0]*x + p[1] - y 
             
 
# Main fitting routine
def MyReg(xf, yf, exf, eyf, a0, b0, x0):
    par0 = a0, b0
    
    # Standard solution
    #res_lsq = least_squares(resid, par0, args=(xf,yf))

    #rsl = least_squares(resid, par0, loss='soft_l1', f_scale=0.5, args=(xf,yf))
    rsl = least_squares(resid, par0, loss='cauchy', f_scale=0.5, args=(xf,yf))
    #rsl = least_squares(resid, par0, method='lm' f_scale=0.5, args=(xf,yf))

    af = rsl.x[0]
    # return zero point at z0
    bf = x0*rsl.x[0] + rsl.x[1]
    #print ("fit= %.3f  %.3f  " % (af, bf))

    # rxy coeff
    lr = linregress(xf, yf)
    rxy0 = lr[2]

    # MonteCarlo for error bars
    aff = np.zeros(100)
    bff = np.zeros(100)
    rxy = np.zeros(100)
    for i in range(100):
        xf1 = np.random.normal(loc=xf, scale=exf)
        yf1 = np.random.normal(loc=yf, scale=eyf)
        rsl1 = least_squares(resid, par0, loss='soft_l1', f_scale=0.1, args=(xf1,yf1))
        aff[i] = rsl1.x[0]
        # return zero point at z0
        bff[i] = x0*rsl1.x[0] + rsl1.x[1]
        # Correlation coefficient
        lrmc = linregress(xf1, yf1)
        rxy[i] = lrmc[2]

    # redefine best fit as median of distribution
    af = np.median(aff)
    # return zero point at z0
    bf = np.median(bff)
    #
    eaf = np.std(aff)
    ebf = np.std(bff)
    #print ("err(%.3f  %.3f)\n" % (eaf, ebf))

    erxy0 = np.std(rxy)

    parf = af, eaf, bf, ebf, rxy0, erxy0
    return parf


def MCFit(r, val):
# This method does not assume any errors
# The bootstrap takes care of it
  xdat = 1.0*r
# Here choose which parameter to fit
# Age
#ydat = 1.0*age
# Metallicity
  ydat = 1.0*val
  ndat = xdat.shape[0]

# Show results in a plot
  plt.scatter(xdat,ydat,color='grey', marker = 'x')
  xmin = np.min(xdat)
  xmax = np.max(xdat)

# Standard least squares
  A = np.vstack([xdat, np.ones(len(xdat))]).T
  a0, b0 = np.linalg.lstsq(A, ydat, rcond=None)[0]
#
  xfit = np.linspace(xmin, xmax, 200)
  yfit0 = a0*xfit + b0
  plt.plot(xfit,yfit0,'k',label='LSQ')

# MonteCarlo results by bootstrapping sample
# select random set comprising 75% of original sample
  nMC = 100
  slp = np.empty(nMC)
  zp = np.empty(nMC)

  nrun = 3*ndat/4
  nrun = int(nrun)
  xMC = np.empty(nrun)
  yMC = np.empty(nrun)

  for iMC in range(nMC):
    idx = np.random.permutation(ndat)
    for i in range(nrun):
      xMC[i] = xdat[idx[i]]
      yMC[i] = ydat[idx[i]]

    A = np.vstack([xMC, np.ones(nrun)]).T
    slp[iMC], zp[iMC] = np.linalg.lstsq(A, yMC, rcond=None)[0]
    
  yfit1 = np.mean(slp) * xfit + np.mean(zp)
  plt.plot(xfit, yfit1, 'r--', label = 'MC')
    
  return a0, np.mean(slp), np.std(slp)

# Calculate simple stats of bootstrapped values to give
# mean, median and stdev
  #print("*** LINEAR FIT  (slope / intercept) ***")
  #print("Best value:  %.6f  /  %6f" % (a0,b0))
  #print("Median:  %.6f  /  %.6f" % (np.median(slp),np.median(zp)))
  #print("Mean:  %.6f  /  %.6f" % (np.mean(slp),np.mean(zp)))
  #print("Standard deviation:  %.6f  /  %.6f" % (np.std(slp),np.std(zp)))

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))



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
  TA = []
  PA = []
  TZ = []
  PZ = []

  #compile saved NNs
  Zmodel.compile(loss='mse',
                  optimizer=tf.train.RMSPropOptimizer(0.001),
                  metrics=['mae', 'mse'])
  Agemodel.compile(loss='mse',
                  optimizer=tf.train.RMSPropOptimizer(0.001),
                  metrics=['mae', 'mse'])
  print("************************************")
  #Iterate over each galaxy
  for phi in gals:
    print(phi)

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

    #Plots of age, Z against r for the galaxy
    plt.clf()
    plt.subplot(221)
    a0, b0, c0 = MCFit(rz, Z)
    plt.ylabel('log(Z$_{spec}$/Z$_{sun})')
    plt.subplot(222)
    a1, b1, c1 = MCFit(rz, zpred)
    plt.ylabel('log(Z$_{pred}$/Z$_{sun})')
    plt.subplot(223)
    a2, b2, c2 = MCFit(ra, age)
    plt.ylabel('log(Age$_{spec}$')
    plt.subplot(224)
    a3, b3, c3 = MCFit(ra, apred)
    plt.ylabel('log(Age$_{pred}$)')

    fname = 'Morphology/' + phi + 'GalGrad70.pdf'
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
  
    plt.clf()
    #Plot of dZ/dr error
    gaussx = np.linspace(min(b0-3*c0, b1 - 3*c1), max(b0+c0*3, b1+3*c1), 1000)
    y0 = gaussian(gaussx, b0, c0)
    y1 = gaussian(gaussx, b1, c1)
    plt.plot(gaussx, y0, 'k',label = 'Spec Z')
    plt.plot(gaussx, y1, 'r', label = 'Predicted Z')
    plt.xlabel('Gradient')
    fname = 'Morphology/' + phi + 'ZCurve70.pdf'
    plt.legend()
    plt.savefig(fname)
  
    plt.clf()
    #dAge/dr error plot
    gaussx = np.linspace(min(b2-3*c2, b3 - 3*c3), max(b2+c2*3, b3+3*c3), 1000)
    y2 = gaussian(gaussx, b2, c2)
    y3 = gaussian(gaussx, b3, c3)
    plt.plot(gaussx, y2, 'k',label = 'Spec age')
    plt.plot(gaussx, y3, 'r', label = 'Predicted age')
    plt.xlabel('Gradient')
    #plt.ylabel('Probability')
    fname = 'Morphology/' + phi + 'AgeCurve70.pdf'
    plt.legend()
    plt.savefig(fname)
  
    plt.clf()
    #True vs pred plot for the galaxy
    plt.subplot(122)
    plt.plot(age, apred, 'kx')
    plt.plot([min(apred), max(apred)], [min(apred), max(apred)], 'r')
    plt.xlabel('log(Age$_{spec}$)')
    plt.ylabel('log(Age$_{CNN}$)')
    plt.subplot(121)
    plt.plot(Z, zpred, 'kx')
    plt.plot([min(zpred), max(zpred)], [min(zpred), max(zpred)], 'r')
    plt.xlabel('log(Z$_{spec}$/Z$_{sun}$)')
    plt.ylabel('log(Z$_{CNN}$/Z$_{sun}$)')
    plt.savefig('Morphology/' + phi + 'TrueVsPred70.pdf')


    dz = b1 - b0#Difference in MC gradient
    da = b3 - b2
    MCzgrad.append(dz)
    MCagrad.append(da)
  
    Aerror=np.sqrt(c2**2 + c3**2)
    ageerror.append(Aerror) #Error in age gradient for this galaxy
    Zerror = np.sqrt(c0**2 + c1**2)
    zerror.append(Zerror) #Error in Z gradient
    
    TA.append(b2)
    PA.append(b3)
    TZ.append(b0)
    PZ.append(b1)

    #lowage = da - Aerror
    #highage = da + Aerror
    #lowzed = dz - Zerror
    #highzed = dz + Zerror
  
    #lowagrad.append(lowage) #Lower (1 sigma) limit for gradient 
    #highagrad.append(highage)
    #lowzgrad.append(lowzed)
    #highzgrad.append(highzed)
  output = dict()  
  output['dapred'] = dapred
  output['datrue'] = datrue
  output['dzpred'] = dzpred
  output['dztrue'] = dztrue
  output['MCagrad'] = MCagrad
  output['MCzgrad'] = MCzgrad
  output['Aerror'] = ageerror
  output['Zerror'] = zerror
  output['trueage'] = TA
  output['predage'] = PA
  output['truez'] = TZ
  output['predz'] = PZ
  output['name'] = gals
  
  #return dapred, datrue, dzpred, dztrue, MCagrad, MCzgrad, Aerror, Zerror
  return output
    
    

#Deltaage is just dapred - datrue
  

