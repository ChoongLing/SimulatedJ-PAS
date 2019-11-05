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
import scipy
from ApplyModel import ApplyModel
from astropy.stats import mad_std


Stats = np.loadtxt('../ChiStats.dat')
mean = Stats[:, 0]
deviation = Stats[:, 1]

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
IZmodel = tf.keras.models.load_model('ZForGroupIChi75.h5')
IAgemodel = tf.keras.models.load_model('AgeForGroupIChi75.h5')
JZmodel = tf.keras.models.load_model('ZForGroupJChi75.h5')
JAgemodel = tf.keras.models.load_model('AgeForGroupJChi75.h5')
KZmodel = tf.keras.models.load_model('ZForGroupKChi75.h5')
KAgemodel = tf.keras.models.load_model('AgeForGroupKChi75.h5')
LZmodel = tf.keras.models.load_model('ZForGroupLChi75.h5')
LAgemodel = tf.keras.models.load_model('AgeForGroupLChi75.h5')


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


allnames = GroupI['name']+ GroupJ['name']+ GroupK['name']+ GroupL['name']
TrueAge = np.concatenate((GroupI['trueage'], GroupJ['trueage'], GroupK['trueage'], GroupL['trueage']))
PredAge = np.concatenate((GroupI['predage'], GroupJ['predage'], GroupK['predage'], GroupL['predage']))
PredZ = np.concatenate((GroupI['predz'], GroupJ['predz'], GroupK['predz'], GroupL['predz']))
TrueZ = np.concatenate((GroupI['truez'], GroupJ['truez'], GroupK['truez'], GroupL['truez']))

#np.save('SetB_grad_age', TrueAge)
#np.save('SetB_grad_z', TrueZ)

plt.rcParams.update({'font.size': 14})

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
lim = 1.#np.ceil(np.abs([MCzgrad, MCagrad]).max() / binwidth) * binwidth
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


plt.savefig('ScatteredHistsSetBChi75.pdf', bbox_inches = 'tight')

plt.rcParams.update({'font.size': 16})

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
plt.savefig('AgeSetBChi75.pdf', bbox_inches = 'tight')

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
plt.savefig('ZSetBChi75.pdf', bbox_inches = 'tight')


deltaage = []
deltaz = []
for i in range(0, len(dztrue)):
  tempa = dapred[i] - datrue[i]
  tempz = dzpred[i] - dztrue[i]
  deltaage.append(tempa)
  deltaz.append(tempz)


devage = mad_std(deltaage)#np.std(deltaage)
devz = mad_std(deltaz,axis = None)#np.std(deltaz)
print('age = ' + str(devage))
print('Z = '+ str(devz))



#posheads = ['name', 'n1', 'n2', 'PA', 'ba', 'kpcarc']
#positions = pd.read_table('../Galaxy_Parameters.txt', delim_whitespace = True, names = posheads)

#inclin = []
#for phi in Igals:
  #Lpos = positions.loc[positions['name'].isin([phi])] 
  #inc = float(Lpos['PA'])
  ##print(inc)
  #inclin.append(inc)
#for phi in Jgals:
  #Lpos = positions.loc[positions['name'].isin([phi])] 
  #inc = float(Lpos['PA'])
  ##print(inc)
  #inclin.append(inc)
#for phi in Kgals:
  #Lpos = positions.loc[positions['name'].isin([phi])] 
  #inc = float(Lpos['PA'])
  ##print(inc)
  #inclin.append(inc)
#for phi in Lgals:
  #Lpos = positions.loc[positions['name'].isin([phi])] 
  #inc = float(Lpos['PA'])
  ##print(inc)
  #inclin.append(inc)

#plt.clf()
#plt.plot(inclin, MCagrad, 'ro', label = 'age')
#plt.plot(inclin, MCzgrad, 'ks', label = 'Z')
#plt.ylim([1, -1])
#plt.xlabel('inclination')
#plt.ylabel('gradient')
#plt.savefig('IncVsGrad.pdf')

agradsig = mad_std(MCagrad)#np.std(MCagrad)#np.std(Asorted)
zgradsig = mad_std(MCzgrad)#np.std(MCzgrad)#np.std(Zsorted)

print('Age gradient dispersion = ' + str(agradsig))
print('Z gradient dispersion = ' + str(zgradsig))

mheads = ['gal', 'mass', 'dunno']
masses = pd.read_table('../masses.dat', delim_whitespace = True, names = mheads)
mass = np.array(masses['mass'])

ellipticallist = open('../Ellipticals.txt')
spirallist = open('../Spirals.txt')
ells = []
for i in ellipticallist:
  split = i.split()
  ells.append(split[0])
spis = []
for i in spirallist:
  split = i.split()
  spis.append(split[0])
  
  
plt.clf()
plt.rcParams.update({'font.size': 16})

lows = []
lowmids = []
himids = []
highs = []
lowe = []
lowmide = []
himide = []
highe = []

for i in range(0, len(PredAge)):
  if allnames[i] in ells:
    plt.scatter(mass[i], PredAge[i], marker = 'o',  facecolor='none', edgecolor='r')
    if mass[i] < 10.:
      lowe.append(PredAge[i])
    elif mass[i] < 10.5:
      lowmide.append(PredAge[i])
    elif mass[i] < 11.:
      himide.append(PredAge[i])
    else:
      highe.append(PredAge[i])
      
  elif allnames[i] in spis:
    plt.scatter(mass[i], PredAge[i], marker ='s', facecolor='none', edgecolor='k')
    if mass[i] < 10.:
      lows.append(PredAge[i])
    elif mass[i] < 10.5:
      lowmids.append(PredAge[i])
    elif mass[i] < 11.:
      himids.append(PredAge[i])
    else:
      highs.append(PredAge[i])

plt.scatter(mass[-7], PredAge[-7], marker = 'o',  facecolor='none', edgecolor='r', label = 'Early Types')
plt.scatter(mass[0], PredAge[0], marker ='s', facecolor='none', edgecolor='k', label = 'Late Types')

#print(lowe, highe)
massbins = np.array([9.5, 10.25, 10.75, 11.25])
ave = []
deve = []
avs = []
devs = []

ave.append(np.median(lowe))
ave.append(np.median(lowmide))
ave.append(np.median(himide))
ave.append(np.median(highe))
deve.append(mad_std(lowe))
deve.append(mad_std(lowmide))
deve.append(mad_std(himide))
deve.append(mad_std(highe))

avs.append(np.median(lows))
avs.append(np.median(lowmids))
avs.append(np.median(himids))
avs.append(np.median(highs))
devs.append(mad_std(lows))
devs.append(mad_std(lowmids))
devs.append(mad_std(himids))
devs.append(mad_std(highs))

plt.errorbar(massbins, ave, yerr = deve, c ='r', marker = 'o', ms = 15, linestyle = ' ')
plt.errorbar(massbins, avs, yerr = devs, c ='k', marker = 's', ms = 15, linestyle = ' ')


plt.xlabel('log(M$_\star$ / M$_\odot$)')
plt.ylabel('grad(Age$_{CNN, B}$/Gyr)/ dex/r$_{eff}$')
plt.ylim([-0.4, 0.4])
plt.legend()
plt.savefig('MassVAgeGradientSetB.pdf')
plt.clf()

dA = []
dZ = []
for i in range(0, len(TrueAge)):
  d = TrueAge[i] - PredAge[i]
  dA.append(d)
  d = TrueZ[i] - PredZ[i]
  dZ.append(d)
  
lows = []
lowmids = []
himids = []
highs = []
lowe = []
lowmide = []
himide = []
highe = []

for i in range(0, len(PredZ)):
  if allnames[i] in ells:
    plt.scatter(mass[i], PredZ[i], marker = 'o',  facecolor='none', edgecolor='r')
    if mass[i] < 10.:
      lowe.append(PredZ[i])
    elif mass[i] < 10.5:
      lowmide.append(PredZ[i])
    elif mass[i] < 11.:
      himide.append(PredZ[i])
    else:
      highe.append(PredZ[i])
      
  elif allnames[i] in spis:
    plt.scatter(mass[i], PredZ[i], marker ='s', facecolor='none', edgecolor='k')
    if mass[i] < 10.:
      lows.append(PredZ[i])
    elif mass[i] < 10.5:
      lowmids.append(PredZ[i])
    elif mass[i] < 11.:
      himids.append(PredZ[i])
    else:
      highs.append(PredZ[i])

plt.scatter(mass[-7], PredZ[-7], marker = 'o',  facecolor='none', edgecolor='r', label = 'Early Types')
plt.scatter(mass[0], PredZ[0], marker ='s', facecolor='none', edgecolor='k', label = 'Late Types')

#print(lowe, highe)
massbins = np.array([9.5, 10.25, 10.75, 11.25])
ave = []
deve = []
avs = []
devs = []

ave.append(np.median(lowe))
ave.append(np.median(lowmide))
ave.append(np.median(himide))
ave.append(np.median(highe))
deve.append(mad_std(lowe))
deve.append(mad_std(lowmide))
deve.append(mad_std(himide))
deve.append(mad_std(highe))

avs.append(np.median(lows))
avs.append(np.median(lowmids))
avs.append(np.median(himids))
avs.append(np.median(highs))
devs.append(mad_std(lows))
devs.append(mad_std(lowmids))
devs.append(mad_std(himids))
devs.append(mad_std(highs))

plt.errorbar(massbins, ave, yerr = deve, c ='r', marker = 'o', ms = 15, linestyle = ' ')
plt.errorbar(massbins, avs, yerr = devs, c ='k', marker = 's', ms = 15, linestyle = ' ')


plt.xlabel('log(M$_\star$ / M$_\odot$)')
plt.ylabel('grad(Z$_{CNN, B}$/Gyr)/ dex/r$_{eff}$')
plt.ylim([-0.4, 0.4])
plt.legend()
plt.savefig('MassVZGradientSetB.pdf')

dA = []
dZ = []
for i in range(0, len(TrueAge)):
  d = TrueAge[i] - PredAge[i]
  dA.append(d)
  d = TrueZ[i] - PredZ[i]
  dZ.append(d)
  
  
  

plt.clf()
plt.scatter(TrueAge, PredAge, label = 'age', facecolor = 'k')
plt.scatter(TrueZ, PredZ, label = 'Z', facecolor = 'r')
plt.xlabel('Spec Gradient (B) ')
plt.ylabel('CNN Gradient (B)')
plt.plot([-1.5, 1.5], [-1.5, 1.5], color = 'grey')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.legend()
plt.savefig('GradsSetB.pdf')