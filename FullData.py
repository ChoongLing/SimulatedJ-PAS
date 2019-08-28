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
import bokeh
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.models import Range1d
from bokeh.models.glyphs import Quad
from ApplyModel import ApplyModel


#def norm(x):
  ##array = np.array(train_stats['std'])
  ##for i in x:
    ##if array[i] == 0.:
      ##return 0.
    ##else:
  #return (x - mean) / deviation


## Define fitting function (linear)
#def lin_func(p, x):
     #a, b = p
     #return a*x + b

#def resid(p, x, y):
    #return p[0]*x + p[1] - y 
             
 
## Main fitting routine
#def MyReg(xf, yf, exf, eyf, a0, b0, x0):
    #par0 = a0, b0
    
    ## Standard solution
    ##res_lsq = least_squares(resid, par0, args=(xf,yf))

    ##rsl = least_squares(resid, par0, loss='soft_l1', f_scale=0.5, args=(xf,yf))
    #rsl = least_squares(resid, par0, loss='cauchy', f_scale=0.5, args=(xf,yf))
    ##rsl = least_squares(resid, par0, method='lm' f_scale=0.5, args=(xf,yf))

    #af = rsl.x[0]
    ## return zero point at z0
    #bf = x0*rsl.x[0] + rsl.x[1]
    ##print ("fit= %.3f  %.3f  " % (af, bf))

    ## rxy coeff
    #lr = linregress(xf, yf)
    #rxy0 = lr[2]

    ## MonteCarlo for error bars
    #aff = np.zeros(100)
    #bff = np.zeros(100)
    #rxy = np.zeros(100)
    #for i in range(100):
        #xf1 = np.random.normal(loc=xf, scale=exf)
        #yf1 = np.random.normal(loc=yf, scale=eyf)
        #rsl1 = least_squares(resid, par0, loss='soft_l1', f_scale=0.1, args=(xf1,yf1))
        #aff[i] = rsl1.x[0]
        ## return zero point at z0
        #bff[i] = x0*rsl1.x[0] + rsl1.x[1]
        ## Correlation coefficient
        #lrmc = linregress(xf1, yf1)
        #rxy[i] = lrmc[2]

    ## redefine best fit as median of distribution
    #af = np.median(aff)
    ## return zero point at z0
    #bf = np.median(bff)
    ##
    #eaf = np.std(aff)
    #ebf = np.std(bff)
    ##print ("err(%.3f  %.3f)\n" % (eaf, ebf))

    #erxy0 = np.std(rxy)

    #parf = af, eaf, bf, ebf, rxy0, erxy0
    #return parf


#def MCFit(r, val):
## This method does not assume any errors
## The bootstrap takes care of it
  #xdat = 1.0*r
## Here choose which parameter to fit
## Age
##ydat = 1.0*age
## Metallicity
  #ydat = val#1.0*val
  #ndat = xdat.shape[0]

## Show results in a plot
  #plt.scatter(xdat,ydat,color='grey', marker = 'x')
  #xmin = np.min(xdat)
  #xmax = np.max(xdat)

## Standard least squares
  #A = np.vstack([xdat, np.ones(len(xdat))]).T
  #a0, b0 = np.linalg.lstsq(A, ydat, rcond=None)[0]
##
  #xfit = np.linspace(xmin, xmax, 200)
  #yfit0 = a0*xfit + b0
  #plt.plot(xfit,yfit0,'k',label='LSQ')

## MonteCarlo results by bootstrapping sample
## select random set comprising 75% of original sample
  #nMC = 100
  #slp = np.empty(nMC)
  #zp = np.empty(nMC)

  #nrun = 3*ndat/4
  #nrun = int(nrun)
  #xMC = np.empty(nrun)
  #yMC = np.empty(nrun)

  #for iMC in range(nMC):
    #idx = np.random.permutation(ndat)
    #for i in range(nrun):
      #xMC[i] = xdat[idx[i]]
      #yMC[i] = ydat[idx[i]]

    #A = np.vstack([xMC, np.ones(nrun)]).T
    #slp[iMC], zp[iMC] = np.linalg.lstsq(A, yMC, rcond=None)[0]
    
  #yfit1 = np.mean(slp) * xfit + np.mean(zp)
  #plt.plot(xfit, yfit1, 'r--', label = 'MC')
    
  #return a0, np.mean(slp), np.std(slp)

## Calculate simple stats of bootstrapped values to give
## mean, median and stdev
  ##print("*** LINEAR FIT  (slope / intercept) ***")
  ##print("Best value:  %.6f  /  %6f" % (a0,b0))
  ##print("Median:  %.6f  /  %.6f" % (np.median(slp),np.median(zp)))
  ##print("Mean:  %.6f  /  %.6f" % (np.mean(slp),np.mean(zp)))
  ##print("Standard deviation:  %.6f  /  %.6f" % (np.std(slp),np.std(zp)))

#def gaussian(x, mu, sig):
    #return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

#Load in the stats for normalisation
stats = np.loadtxt('../ChiStats.dat')
mean = stats[:, 0]
deviation = stats[:, 1]

rnames = ['Galaxy', 'arcsec', 'Reff']
Reff=pd.read_table('../reff.dat', delim_whitespace = True, names = rnames)

#Import data
column_names = ['name', 'age', 'ZHL', 'JPAS3900_145', 'JPAS4000_145', \
  'JPAS4100_145', 'JPAS4200_145', 'JPAS4300_145', 'JPAS4400_145', 'JPAS4500_145', 'JPAS4600_145', 'JPAS4700_145',\
  'JPAS4800_145', 'JPAS4900_145', 'JPAS5000_145', 'JPAS5100_145', 'JPAS5200_145', 'JPAS5300_145', 'JPAS5400_145',\
  'JPAS5500_145', 'JPAS5600_145', 'JPAS5700_145', 'JPAS5800_145', 'JPAS5900_145', 'JPAS6000_145', 'JPAS6100_145',\
  'JPAS6200_145', 'JPAS6300_145', 'JPAS6400_145', 'JPAS6500_145', 'JPAS6600_145', 'JPAS6700_145', 'JPAS6800_145', \
  'JPAS6900_145', 'JPAS7000_145', 'JPAS7100_145', 'JPAS7200_145', 'JPAS7300_145', 'xkpc', 'ykpc']

feature_names = column_names[0] #column for metallicity
label_names = column_names[1:] #columns for the filters

incoming = pd.read_table('../LowChi.dat', delim_whitespace = True, names = column_names)
#Masking H_alpha
incoming = incoming.drop(columns = 'JPAS6600_145')
incoming = incoming.drop(columns = 'JPAS6700_145')
dataset = incoming.copy()
dataset = dataset.dropna()
#print(len(dataset.T))

galx = open('NewI.txt', 'r')
Igals = []
for i in galx:
  split = i.split()
  Igals.append(split[0])
galx = open('../GroupJ/NewJ.txt', 'r')
Jgals = []
for i in galx:
  split = i.split()
  Jgals.append(split[0])
galx = open('../GroupK/NewK.txt', 'r')
Kgals = []
for i in galx:
  split = i.split()
  Kgals.append(split[0])
galx = open('../GroupL/NewL.txt', 'r')
Lgals = []
for i in galx:
  split = i.split()
  Lgals.append(split[0])


#Load saved NNs
IZmodel = tf.keras.models.load_model('ZForGroupIChi70.h5')
IAgemodel = tf.keras.models.load_model('AgeForGroupIChi70.h5')
JZmodel = tf.keras.models.load_model('ZForGroupJChi70.h5')
JAgemodel = tf.keras.models.load_model('AgeForGroupJChi70.h5')
KZmodel = tf.keras.models.load_model('ZForGroupKChi70.h5')
KAgemodel = tf.keras.models.load_model('AgeForGroupKChi70.h5')
LZmodel = tf.keras.models.load_model('ZForGroupLChi70.h5')
LAgemodel = tf.keras.models.load_model('AgeForGroupLChi70.h5')


#Idapred, Idatrue, Idzpred, Idztrue, IMCagrad, IMCzgrad, IAerror, IZerror 
GroupI = ApplyModel(IAgemodel, IZmodel, dataset, Igals)
  
#Jdapred, Jdatrue, Jdzpred, Jdztrue, JMCagrad, JMCzgrad, JAerror, JZerror
GroupJ = ApplyModel(JAgemodel, JZmodel, dataset, Jgals)
  
#Kdapred, Kdatrue, Kdzpred, Kdztrue, KMCagrad, KMCzgrad, KAerror, KZerror
GroupK = ApplyModel(KAgemodel, KZmodel, dataset, Kgals)
  
#Ldapred, Ldatrue, Ldzpred, Ldztrue, LMCagrad, LMCzgrad, LAerror, LZerror
GroupL = ApplyModel(LAgemodel, LZmodel, dataset, Lgals)
print(len(GroupL['MCagrad']),len(GroupL['Aerror']))

dapred = np.concatenate((GroupI['dapred'], GroupJ['dapred'], GroupK['dapred'], GroupL['dapred']))
datrue = np.concatenate((GroupI['datrue'], GroupJ['datrue'], GroupK['datrue'], GroupL['datrue']))
dzpred = np.concatenate((GroupI['dzpred'], GroupJ['dzpred'], GroupK['dzpred'], GroupL['dzpred']))
dztrue = np.concatenate((GroupI['dztrue'], GroupJ['dztrue'], GroupK['dztrue'], GroupL['dztrue']))



MCagrad = np.concatenate((GroupI['MCagrad'], GroupJ['MCagrad'], GroupK['MCagrad'], GroupL['MCagrad']))
MCzgrad = np.concatenate((GroupI['MCzgrad'], GroupJ['MCzgrad'], GroupK['MCzgrad'], GroupL['MCzgrad']))
ageerror = np.concatenate((GroupI['Aerror'], GroupJ['Aerror'], GroupK['Aerror'], GroupL['Aerror']))
zerror = np.concatenate((GroupI['Zerror'], GroupJ['Zerror'], GroupK['Zerror'], GroupL['Zerror']))


plt.clf()
#Plot a scatter plot with histograms
# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005
rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]
# start with a rectangular Figure
plt.figure(figsize=(8, 8))
ax_scatter = plt.axes(rect_scatter)
ax_scatter.tick_params(direction='in', top=True, right=True)
ax_histx = plt.axes(rect_histx)
ax_histx.tick_params(direction='in', labelbottom=False)
ax_histy = plt.axes(rect_histy)
ax_histy.tick_params(direction='in', labelleft=False)
# the scatter plot:
ax_scatter.errorbar(MCzgrad, MCagrad,  yerr = ageerror, xerr = zerror, fmt = 'kx', ecolor = 'r')

# now determine nice limits by hand:
binwidth = 0.05
lim = 1.5#np.ceil(np.abs([MCzgrad, MCagrad]).max() / binwidth) * binwidth
ax_scatter.set_xlim((-lim, lim))
ax_scatter.set_ylim((-lim, lim))
ax_scatter.set_xlabel('grad(Z$_{CNN}$) - grad(Z$_{spec}$)')
ax_scatter.xaxis.label.set_fontsize(14)
ax_scatter.set_ylabel('grad(Age$_{CNN}$) - grad(Age$_{spec}$)')
ax_scatter.yaxis.label.set_fontsize(14)
ax_scatter.set_xticks(ax_scatter.get_xticks()[:-1])
ax_scatter.set_yticks(ax_scatter.get_yticks()[:-1])

bins = np.arange(-lim, lim + binwidth, binwidth)
ax_histx.hist(MCzgrad, bins=bins, color = 'k')
ax_histx.set_ylabel('Count')
ax_histx.yaxis.label.set_fontsize(14)
ax_histy.hist(MCagrad, bins=bins, orientation='horizontal', color = 'k')
ax_histy.set_xlabel('Count')
ax_histy.xaxis.label.set_fontsize(14)


plt.savefig('ScatteredHistsAllSetBChi70.pdf')

plt.rcParams.update({'font.size': 14})

datrue = np.array(datrue)
dztrue = np.array(dztrue)
dapred = np.array(dapred)
dzpred = np.array(dzpred)

#Contour plot of age
plt.clf()
nbins=300
k = kde.gaussian_kde([datrue,dapred])
xi, yi = np.mgrid[datrue.min():datrue.max():nbins*1j, dapred.min():dapred.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
plt.scatter(datrue, dapred, marker='.', color = 'grey', alpha = 0.2)
plt.plot([8.25, 10.3], [8.25, 10.3], 'k')
plt.contour(xi, yi, zi.reshape(xi.shape), cmap = 'hot')
plt.xlim(9.2, 10.3)
plt.ylim(9.2, 10.3)
plt.xlabel('log(Age$_{spec})$')
plt.ylabel('log(Age$_{CNN})$')
plt.savefig('AgeAllSetB70.pdf', bbox_inches = 'tight')

#Contour plot of Z
plt.clf()
k = kde.gaussian_kde([dztrue,dzpred])
xi, yi = np.mgrid[dztrue.min():dztrue.max():nbins*1j, dzpred.min():dzpred.max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
plt.scatter(dztrue, dzpred, marker='.', color = 'grey', alpha = 0.2)
plt.plot([-1.25, 0.25], [-1.25, 0.25], 'k')
plt.contour(xi, yi, zi.reshape(xi.shape), cmap = 'hot')
plt.xlim(-0.65, 0.25)
plt.ylim(-0.65, 0.25)
plt.xlabel('log(Z$_{spec}$/Z$_{sun}$)')
plt.ylabel('log(Z$_{CNN}$/Z$_{sun}$)')
plt.savefig('ZAllSetB70.pdf', bbox_inches = 'tight')



