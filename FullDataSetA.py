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
import astropy.stats as stats
import bokeh
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.models import Range1d
from bokeh.models.glyphs import Quad
from ApplyModelSetA import ApplyModel

def norm(x):
  #array = np.array(train_stats['std'])
  #for i in x:
    #if array[i] == 0.:
      #return 0.
    #else:
  return (x - mean) / deviation


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
  ydat = val#1.0*val
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

longlist = open('NewGals.txt', 'r')

gals = []
for i in longlist:
  split = i.split()
  gals. append(split[0])

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

incoming = pd.read_table('Chi1.dat', delim_whitespace = True, names = column_names)
incoming = incoming.drop(columns = 'JPAS6600_145')
incoming = incoming.drop(columns = 'JPAS6700_145')
dataset = incoming.copy()
dataset1 = dataset.dropna()
incoming = pd.read_table('../Group2/Chi2.dat', delim_whitespace = True, names = column_names)
incoming = incoming.drop(columns = 'JPAS6600_145')
incoming = incoming.drop(columns = 'JPAS6700_145')
dataset = incoming.copy()
dataset2 = dataset.dropna()
incoming = pd.read_table('../Group3/Chi3.dat', delim_whitespace = True, names = column_names)
incoming = incoming.drop(columns = 'JPAS6600_145')
incoming = incoming.drop(columns = 'JPAS6700_145')
dataset = incoming.copy()
dataset3 = dataset.dropna()
incoming = pd.read_table('../Group4/Chi4.dat', delim_whitespace = True, names = column_names)
incoming = incoming.drop(columns = 'JPAS6600_145')
incoming = incoming.drop(columns = 'JPAS6700_145')
dataset = incoming.copy()
dataset4 = dataset.dropna()

Zmodel1 = tf.keras.models.load_model('ZForGroup1Chi75.h5')
Agemodel1 = tf.keras.models.load_model('AgeForGroup1Chi75.h5')
Zmodel2 = tf.keras.models.load_model('ZForGroup2Chi75.h5')
Agemodel2 = tf.keras.models.load_model('AgeForGroup2Chi75.h5')
Zmodel3 = tf.keras.models.load_model('ZForGroup3Chi75.h5')
Agemodel3 = tf.keras.models.load_model('AgeForGroup3Chi75.h5')
Zmodel4 = tf.keras.models.load_model('ZForGroup4Chi75.h5')
Agemodel4 = tf.keras.models.load_model('AgeForGroup4Chi75.h5')

Group1 = ApplyModel(Agemodel1, Zmodel1, dataset1, gals) 
Group2 = ApplyModel(Agemodel2, Zmodel2, dataset2, gals) 
Group3 = ApplyModel(Agemodel3, Zmodel3, dataset3, gals) 
Group4 = ApplyModel(Agemodel4, Zmodel4, dataset4, gals) 
print('done')

analysed = pd.concat([Group1, Group2, Group3, Group4])
plt.rcParams.update({'font.size': 16})

nbins=300
k = kde.gaussian_kde([analysed['datrue'],analysed['dapred']])
xi, yi = np.mgrid[analysed['datrue'].min():analysed['datrue'].max():nbins*1j,
		  analysed['dapred'].min():analysed['dapred'].max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
zi = zi/max(zi)
 
# Make the plot
plt.scatter(analysed['datrue'], analysed['dapred'], marker='.', color = 'grey', alpha = 0.2)
plt.plot([8.25, 10.3], [8.25, 10.3], 'k')
plt.contour(xi, yi, zi.reshape(xi.shape), cmap = 'hot', vmin = 0., vmax = 1.)
plt.xlabel('log(Age$_{spec}$)')
plt.ylabel('log(Age$_{CNN}$)')
plt.xlim(9.2, 10.3)
plt.ylim(9.2, 10.3)
plt.colorbar()
plt.savefig('AgeAllChi75.pdf', bbox_inches = 'tight')


plt.clf()
k = kde.gaussian_kde([analysed['dztrue'],analysed['dzpred']])
xi, yi = np.mgrid[analysed['dztrue'].min():analysed['dztrue'].max():nbins*1j,
		  analysed['dzpred'].min():analysed['dzpred'].max():nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
zi = zi/max(zi)
 
# Make the plot
plt.scatter(analysed['dztrue'], analysed['dzpred'], marker='.', color = 'grey', alpha = 0.2)
plt.plot([-1.25, 0.25], [-1.25, 0.25], 'k')
plt.contour(xi, yi, zi.reshape(xi.shape), cmap = 'hot')
plt.xlabel('log(Z$_{spec}$/Z$_{sun}$)')
plt.ylabel('log(Z$_{CNN}$/Z$_{sun}$)')
plt.xlim(-0.65, 0.25)
plt.ylim(-0.65, 0.25)
plt.colorbar()

plt.savefig('ZAllChi75.pdf', bbox_inches = 'tight')

aslopeerror = []
zslopeerror = []
dzgrad = []
dagrad = []
MCzgrad = []
MCagrad = []
ageerror = []
zerror = []
dzpred = np.array(analysed['dzpred'])
dapred = np.array(analysed['dapred'])
dztrue = np.array(analysed['dztrue'])
datrue = np.array(analysed['datrue'])
TA = []
PA = []
TZ = []
PZ = []

for phi in gals:
  print(phi)
  thisr = Reff.loc[Reff['Galaxy'].isin([phi])]
  reff=  thisr['Reff'].values
  
  thisgal = analysed.loc[analysed['name'].isin([phi])]
  
  Z = np.array(thisgal['dztrue'])
  zpred = np.array(thisgal['dzpred'])
  rz = np.array(thisgal['RZ'])
  age = np.array(thisgal['datrue'])
  apred = np.array(thisgal['dapred'])
  ra = np.array(thisgal['RA'])


  empty = []
  #print(reff)
  if bool(reff) == False:
    print('empty')
    pass
  else:

    plt.clf()
    plt.subplot(221)
    a0, b0, c0 = MCFit(rz, Z) #a is intercept, b is gradient and c is error in gradient
    plt.ylabel('log(Z$_{spec}$/Z$_{sun}$)')
    plt.ylim(-0.1, 0.2)
    plt.xlabel('r/r$_{eff}$')
    #print(a0)
    plt.subplot(222)
    a1, b1, c1 = MCFit(rz, zpred)
    plt.ylabel('log(Z$_{CNN}$/Z$_{sun}$)')
    plt.ylim(-0.1, 0.2)
    plt.xlabel('r/r$_{eff}$')
    plt.subplot(223)
    a2, b2, c2 = MCFit(ra, age)
    plt.ylabel('log(Age$_{spec}$)')
    plt.ylim(9.9, 10.25)
    plt.xlabel('r/r$_{eff}$')
    plt.subplot(224)
    a3, b3, c3 = MCFit(ra, apred)
    plt.ylabel('log(Age$_{CNN}$)')
    plt.ylim(9.9, 10.25)
    plt.xlabel('r/r$_{eff}$')
    fname = 'FullDataset/' + phi + 'GalGrad75.pdf'
    plt.tight_layout()
    plt.savefig(fname, bbox_inches = 'tight')
    
    tempa = np.sqrt(c1**2 + c3**2)
    tempz = np.sqrt(c0**2 + c2**2)
    aslopeerror.append(tempa)
    zslopeerror.append(tempz)
    
    plt.clf()
    gaussx = np.linspace(min(b0-3*c0, b1 - 3*c1), max(b0+c0*3, b1+3*c1), 1000)
    y0 = gaussian(gaussx, b0, c0)
    y1 = gaussian(gaussx, b1, c1)
    plt.plot(gaussx, y0, 'k',label = 'Spectroscopic Z')
    plt.plot(gaussx, y1, 'r', label = 'CNN Z')
    #plt.axvline(a0, 'k--')
    #plt.axvline(a1, 'r--')
    plt.xlabel('Gradient')
    fname = 'FullDataset/' + phi + 'ZCurve75.pdf'
    plt.legend()
    plt.savefig(fname)
  
    plt.clf()
    gaussx = np.linspace(min(b2-3*c2, b3 - 3*c3), max(b2+c2*3, b3+3*c3), 1000)
    y2 = gaussian(gaussx, b2, c2)
    y3 = gaussian(gaussx, b3, c3)
    plt.plot(gaussx, y2, 'k',label = 'Spectroscopic age')
    plt.plot(gaussx, y3, 'r', label = 'CNN age')
    #plt.axvline(a2, 'k--')
    #plt.axvline(a3, 'r--')
    plt.xlabel('Gradient')
    #plt.ylabel('Probability')
    fname = 'FullDataset/' + phi + 'AgeCurve75.pdf'
    plt.legend()
    plt.savefig(fname)
  
    plt.clf()
    plt.subplot(122)
    plt.plot(age, apred, 'kx')
    plt.plot([min(apred), max(apred)], [min(apred), max(apred)], 'r')
    plt.xlabel('log(Age$_{spec}$/Gyr)')
    plt.ylabel('log(Age$_{CNN}$/Gyr)')
    plt.subplot(121)
    plt.plot(Z, zpred, 'kx')
    plt.plot([min(zpred), max(zpred)], [min(zpred), max(zpred)], 'r')
    plt.xlabel('log(Z$_{spec}$/Z$_{sun}$)')
    plt.ylabel('log(Z$_{CNN}$/Z$_{sun}$)')
    plt.savefig('FullDataset/' + phi + 'TrueVsPred75.pdf')

    dz = a1 - a0
    da = a3 - a2
    dzgrad.append(dz)
    dagrad.append(da)
    dz = b1 - b0
    da = b3 - b2
    MCzgrad.append(dz)
    MCagrad.append(da)
  
    Aerror=np.sqrt(c2**2 + c3**2)
    ageerror.append(Aerror)
    Zerror = np.sqrt(c0**2 + c1**2)
    zerror.append(Zerror)
  
    #lowage = da - Aerror
    #highage = da + Aerror
    #lowzed = dz - Zerror
    #highzed = dz + Zerror
  
    TA.append(b2)
    PA.append(b3)
    TZ.append(b0)
    PZ.append(b1)


deltaz = []#dzpred - dztrue
deltaage = []#dapred - datrue
for i in range (0, len(dzpred)):
  tempZ = (dzpred[i] - dztrue[i])
  tempage = (dapred[i] - datrue[i])
  deltaz.append(tempZ)
  deltaage.append(tempage)
  
histz, edgez = np.histogram(deltaz, bins = 30)
hista, edgea = np.histogram(deltaage, bins = 30)

histz = np.array(histz)
edgez = np.array(edgez)


fit = np.polyfit(dzpred, dztrue, 1)
slopez = fit[0]
intercept = fit[1]
xfitz = [min(dzpred), max(dztrue)]
yfitz = [slopez * xx + intercept for xx in xfitz]

fit = np.polyfit(dapred, datrue, 1)
slopea = fit[0]
intercept = fit[1]
xfita = [min(dapred), max(datrue)]
yfita = [slopea * xx + intercept for xx in xfita]

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

plt.rcParams.update({'font.size': 14})

Zsorted = np.sort(MCzgrad)
Asorted = np.sort(MCagrad)
Zsorted = Zsorted[4:-3]
Asorted = Asorted[4:-3]

# now determine nice limits by hand:
binwidth = 0.05
lim = 1.#
ax_scatter.set_xlim((-lim, lim))
ax_scatter.set_ylim((-lim, lim))
ax_scatter.set_xlabel('grad(Z$_{CNN}$) - grad(Z$_{spec}$)')
#ax_scatter.xaxis.label.set_fontsize(14)
ax_scatter.set_ylabel('grad(Age$_{CNN}$) - grad(Age$_{spec}$)')
bins = np.arange(-lim, lim + binwidth, binwidth)
ax_histx.hist(MCzgrad, bins=bins, color = 'k')
ax_histx.set_ylabel('Count')
ax_histy.hist(MCagrad, bins=bins, orientation='horizontal', color = 'k')
ax_histy.set_xlabel('Count')
ax_scatter.set_xticks(ax_scatter.get_xticks()[:-1])
ax_scatter.set_yticks(ax_scatter.get_yticks()[:-1])

ax_histx.set_xlim(ax_scatter.get_xlim())
ax_histy.set_ylim(ax_scatter.get_ylim())


plt.savefig('ScatteredHistsAllChi75.pdf', bbox_inches='tight')

plt.clf()

devage = stats.mad_std(deltaage)#np.std(deltaage)
devz = stats.mad_std(deltaz,axis = None)#np.std(deltaz)
print('age = ' + str(devage))
print('Z = '+ str(devz))


#posheads = ['name', 'n1', 'n2', 'PA', 'ba', 'kpcarc']
#positions = pd.read_table('../Galaxy_Parameters.txt', delim_whitespace = True, names = posheads)

#inclin = []
#for phi in gals:
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
#plt.savefig('IncVsGradSetA.pdf')

agradsig = stats.mad_std(MCagrad)#np.std(MCagrad)#np.std(Asorted)
zgradsig = stats.mad_std(MCzgrad)#np.std(MCzgrad)#np.std(Zsorted)

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

for i in range(0, len(PZ)):
  if gals[i] in ells:
    plt.scatter(mass[i], PZ[i], marker = 'o',  facecolor='none', edgecolor='r')
    if mass[i] < 10.:
      lowe.append(PZ[i])
    elif mass[i] < 10.5:
      lowmide.append(PZ[i])
    elif mass[i] < 11.:
      himide.append(PZ[i])
    else:
      highe.append(PZ[i])
      
  elif gals[i] in spis:
    plt.scatter(mass[i], PZ[i], marker ='s', facecolor='none', edgecolor='k')
    if mass[i] < 10.:
      lows.append(PZ[i])
    elif mass[i] < 10.5:
      lowmids.append(PZ[i])
    elif mass[i] < 11.:
      himids.append(PZ[i])
    else:
      highs.append(PZ[i])

plt.scatter(mass[-7], PZ[-7], marker = 'o',  facecolor='none', edgecolor='r', label = 'Early Types')
plt.scatter(mass[0], PZ[0], marker ='s', facecolor='none', edgecolor='k', label = 'Late Types')

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
deve.append(stats.mad_std(lowe))
deve.append(stats.mad_std(lowmide))
deve.append(stats.mad_std(himide))
deve.append(stats.mad_std(highe))

avs.append(np.median(lows))
avs.append(np.median(lowmids))
avs.append(np.median(himids))
avs.append(np.median(highs))
devs.append(stats.mad_std(lows))
devs.append(stats.mad_std(lowmids))
devs.append(stats.mad_std(himids))
devs.append(stats.mad_std(highs))

plt.errorbar(massbins, avs, yerr = devs, c ='k', marker = 's', ms = 15, linestyle = ' ')
plt.errorbar(massbins, ave, yerr = deve, c ='r', marker = 'o', ms = 15, linestyle = ' ')


plt.xlabel('log(M$_\star$ / M$_\odot$)')
plt.ylabel('grad(Z$_{CNN}$/Gyr)/ dex/r$_{eff}$')
plt.ylim([-0.4, 0.4])
plt.legend()
plt.savefig('MassVZGradientSetA.pdf')
plt.clf()

for i in range(0, len(TZ)):
  if gals[i] in ells:
    plt.scatter(mass[i], TZ[i], marker = 'o',  facecolor='none', edgecolor='r')
    if mass[i] < 10.:
      lowe.append(TZ[i])
    elif mass[i] < 10.5:
      lowmide.append(TZ[i])
    elif mass[i] < 11.:
      himide.append(TZ[i])
    else:
      highe.append(TZ[i])
      
  elif gals[i] in spis:
    plt.scatter(mass[i], TZ[i], marker ='s', facecolor='none', edgecolor='k')
    if mass[i] < 10.:
      lows.append(TZ[i])
    elif mass[i] < 10.5:
      lowmids.append(TZ[i])
    elif mass[i] < 11.:
      himids.append(TZ[i])
    else:
      highs.append(TZ[i])

plt.scatter(mass[-7], TZ[-7], marker = 'o',  facecolor='none', edgecolor='r', label = 'Early Types')
plt.scatter(mass[0], TZ[0], marker ='s', facecolor='none', edgecolor='k', label = 'Late Types')

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
deve.append(stats.mad_std(lowe))
deve.append(stats.mad_std(lowmide))
deve.append(stats.mad_std(himide))
deve.append(stats.mad_std(highe))

avs.append(np.median(lows))
avs.append(np.median(lowmids))
avs.append(np.median(himids))
avs.append(np.median(highs))
devs.append(stats.mad_std(lows))
devs.append(stats.mad_std(lowmids))
devs.append(stats.mad_std(himids))
devs.append(stats.mad_std(highs))

plt.errorbar(massbins, avs, yerr = devs, c ='k', marker = 's', ms = 15, linestyle = ' ')
plt.errorbar(massbins, ave, yerr = deve, c ='r', marker = 'o', ms = 15, linestyle = ' ')


plt.xlabel('log(M$_\star$ / M$_\odot$)')
plt.ylabel('grad(Z$_{spec}$/Gyr)/ dex/r$_{eff}$')
plt.ylim([-0.4, 0.4])
plt.legend()
plt.savefig('SpecMassVZGradientSetA.pdf')
plt.clf()

for i in range(0, len(PA)):
  if gals[i] in ells:
    plt.scatter(mass[i], PA[i], marker = 'o',  facecolor='none', edgecolor='r')
    if mass[i] < 10.:
      lowe.append(PA[i])
    elif mass[i] < 10.5:
      lowmide.append(PA[i])
    elif mass[i] < 11.:
      himide.append(PA[i])
    else:
      highe.append(PA[i])
      
  elif gals[i] in spis:
    plt.scatter(mass[i], PA[i], marker ='s', facecolor='none', edgecolor='k')
    if mass[i] < 10.:
      lows.append(PA[i])
    elif mass[i] < 10.5:
      lowmids.append(PA[i])
    elif mass[i] < 11.:
      himids.append(PA[i])
    else:
      highs.append(PA[i])

plt.scatter(mass[-7], PA[-7], marker = 'o',  facecolor='none', edgecolor='r', label = 'Early Types')
plt.scatter(mass[0], PA[0], marker ='s', facecolor='none', edgecolor='k', label = 'Late Types')

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
deve.append(stats.mad_std(lowe))
deve.append(stats.mad_std(lowmide))
deve.append(stats.mad_std(himide))
deve.append(stats.mad_std(highe))

avs.append(np.median(lows))
avs.append(np.median(lowmids))
avs.append(np.median(himids))
avs.append(np.median(highs))
devs.append(stats.mad_std(lows))
devs.append(stats.mad_std(lowmids))
devs.append(stats.mad_std(himids))
devs.append(stats.mad_std(highs))

plt.errorbar(massbins, avs, yerr = devs, c ='k', marker = 's', ms = 15, linestyle = ' ')
plt.errorbar(massbins, ave, yerr = deve, c ='r', marker = 'o', ms = 15, linestyle = ' ')


plt.xlabel('log(M$_\star$ / M$_\odot$)')
plt.ylabel('grad(Age$_{CNN}$/Gyr)/ dex/r$_{eff}$')
plt.ylim([-0.4, 0.4])
plt.legend()
plt.savefig('MassVAgeGradientSetA.pdf')
plt.clf()

for i in range(0, len(TA)):
  if gals[i] in ells:
    plt.scatter(mass[i], TA[i], marker = 'o',  facecolor='none', edgecolor='r')
    if mass[i] < 10.:
      lowe.append(TA[i])
    elif mass[i] < 10.5:
      lowmide.append(TA[i])
    elif mass[i] < 11.:
      himide.append(TA[i])
    else:
      highe.append(TA[i])
      
  elif gals[i] in spis:
    plt.scatter(mass[i], TA[i], marker ='s', facecolor='none', edgecolor='k')
    if mass[i] < 10.:
      lows.append(TA[i])
    elif mass[i] < 10.5:
      lowmids.append(TA[i])
    elif mass[i] < 11.:
      himids.append(TA[i])
    else:
      highs.append(TA[i])

plt.scatter(mass[-7], TA[-7], marker = 'o',  facecolor='none', edgecolor='r', label = 'Early Types')
plt.scatter(mass[0], TA[0], marker ='s', facecolor='none', edgecolor='k', label = 'Late Types')

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
deve.append(stats.mad_std(lowe))
deve.append(stats.mad_std(lowmide))
deve.append(stats.mad_std(himide))
deve.append(stats.mad_std(highe))

avs.append(np.median(lows))
avs.append(np.median(lowmids))
avs.append(np.median(himids))
avs.append(np.median(highs))
devs.append(stats.mad_std(lows))
devs.append(stats.mad_std(lowmids))
devs.append(stats.mad_std(himids))
devs.append(stats.mad_std(highs))

plt.errorbar(massbins, avs, yerr = devs, c ='k', marker = 's', ms = 15, linestyle = ' ')
plt.errorbar(massbins, ave, yerr = deve, c ='r', marker = 'o', ms = 15, linestyle = ' ')


plt.xlabel('log(M$_\star$ / M$_\odot$)')
plt.ylabel('grad(Age$_{spec}$/Gyr)/ dex/r$_{eff}$')
plt.ylim([-0.4, 0.4])
plt.legend()
plt.savefig('SpecMassVAgeGradientSetA.pdf')


dA = []
dZ = []
for i in range(0, len(TA)):
  d = TA[i] - PA[i]
  dA.append(d)
  d = TZ[i] - PZ[i]
  dZ.append(d)
  


plt.clf()
plt.scatter(TA, PA, label = 'age', facecolor = 'k')
plt.scatter(TZ, PZ, label = 'Z', facecolor = 'r')
plt.xlabel('Spec Gradient (A)')
plt.ylabel('Gradient difference')
plt.xlim([-1.5, 1.5])
plt.plot([-1.5, 1.5], [-1.5, 1.5], color = 'grey')
plt.ylim([-1.5, 1.5])
plt.legend()
plt.savefig('GradsSetA.pdf')