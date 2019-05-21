
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
import linfit
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

#Normalisation function
def norm(x):
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
    #Alternative solutions
    #rsl = least_squares(resid, par0, loss='soft_l1', f_scale=0.5, args=(xf,yf))
    rsl = least_squares(resid, par0, loss='cauchy', f_scale=0.5, args=(xf,yf))
    #rsl = least_squares(resid, par0, method='lm' f_scale=0.5, args=(xf,yf))

    af = rsl.x[0]
    # return zero point at z0
    bf = x0*rsl.x[0] + rsl.x[1]
    print ("fit= %.3f  %.3f  " % (af, bf))

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
  #Age
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


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# Arrays for creating summaries
dztrue = []
dzpred = []
datrue = []
dapred = []
dzgrad = []
dagrad = []
MCzgrad = []
MCagrad = []
lowagrad = []
lowzgrad = []
highagrad = []
highzgrad = []

#Load in the stats for normalisation
stats = np.loadtxt('../Zstats.dat')
mean = stats[:, 0]
deviation = stats[:, 1]

rnames = ['Galaxy', 'h', 'Reff']
Reff=pd.read_table('../reff.dat', delim_whitespace = True, names = rnames)

#Import data
column_names = ['name', 'age', 'ZHL', 'JPAS4000_145', \
  'JPAS4100_145', 'JPAS4200_145', 'JPAS4300_145', 'JPAS4400_145', 'JPAS4500_145', 'JPAS4600_145', 'JPAS4700_145',\
  'JPAS4800_145', 'JPAS4900_145', 'JPAS5000_145', 'JPAS5100_145', 'JPAS5200_145', 'JPAS5300_145', 'JPAS5400_145',\
  'JPAS5500_145','JPAS5600_145', 'JPAS5700_145', 'JPAS5800_145', 'JPAS5900_145', 'JPAS6000_145', 'JPAS6100_145',\
  'JPAS6200_145', 'JPAS6300_145', 'JPAS6400_145', 'JPAS6500_145', 'JPAS6600_145', 'JPAS6700_145', 'JPAS6800_145', \
  'JPAS6900_145', 'JPAS7000_145', 'JPAS7100_145', 'JPAS7200_145', 'JPAS7300_145', 'xkpc', 'ykpc']

feature_names = column_names[0] #column for metallicity
label_names = column_names[1:] #columns for the filters

incoming = pd.read_table('../Alldata.dat', delim_whitespace = True, names = column_names)
#Masking H_alpha
incoming = incoming.drop(columns = 'JPAS6600_145')
incoming = incoming.drop(columns = 'JPAS6700_145')
dataset = incoming.copy()
dataset = dataset.dropna()



#Load list of galaxies in this group
galx = open('UnbrokenL.txt', 'r')
Lgals = []
for i in galx:
  split = i.split()
  Lgals.append(split[0])


#Load and compile saved NNs
Zmodel = tf.keras.models.load_model('ZForGroupLUnbroken.h5')
Agemodel = tf.keras.models.load_model('AgeForGroupLUnbroken.h5')
Zmodel.compile(loss='mse',
                optimizer=tf.train.RMSPropOptimizer(0.001),
                metrics=['mae', 'mse'])
Agemodel.compile(loss='mse',
                optimizer=tf.train.RMSPropOptimizer(0.001),
                metrics=['mae', 'mse'])


for phi in Lgals: #Iterate fitting over each galaxy
  print(phi)

  galaxy = dataset.loc[dataset['name'].isin([phi])]# Dataset for galaxy
  galaxy = galaxy.drop(columns = 'name')
  
  thisr = Reff.loc[Reff['Galaxy'].isin([phi])] #Get effective raius for the galaxy
  reff=  thisr['Reff'].values

  smaller = galaxy.sample(frac = 1, random_state = 0) #Reduction of dataset size
  #smaller = smaller1[incoming.ZHL > -0.75] # metallicity

  
  
  #Running the model on Z *****************************************
  zdataset = smaller.copy()
  zdataset = zdataset.dropna()
  zdataset = zdataset.drop(columns = 'age')
  
  #Divide testing set into two
  ztrain_set2 = zdataset.sample(frac = 0.5, random_state = 0)
  ztest_set2 = zdataset.drop(ztrain_set2.index)
  
  #Preserve a copy of the datasets to get positions
  ztrain_positions = ztrain_set2.copy()
  ztest_positions = ztest_set2.copy()

  ztrain_set2 = ztrain_set2.drop(columns = 'xkpc')
  ztrain_set2 = ztrain_set2.drop(columns = 'ykpc')
  ztest_set2 = ztest_set2.drop(columns = 'xkpc')
  ztest_set2 = ztest_set2.drop(columns = 'ykpc')

  #Ture values for Z
  ztestlabel2 = ztest_set2.pop('ZHL')
  ztrainlabel2 = ztrain_set2.pop('ZHL')

  #Normalisation
  znormtest = norm(ztest_set2)
  znormtrain = norm(ztrain_set2)
  
  #Predict Z using NN model 
  zpredtest = Zmodel.predict(znormtest)
  zpredtrain = Zmodel.predict(znormtrain)

  #Join the sets of two arrays back into one
  zpred = np.concatenate((zpredtrain, zpredtest), axis = 0) 
  zpred = zpred.flatten()


  Z = np.concatenate((ztrainlabel2, ztestlabel2), axis = 0).flatten()#flux[:, 1] #True values for Z

  #Min and max values for Z from CALIFA; used as a boundary for predicted values
  Zmax = 0.21999
  Zmin = -1.30969
  for i in range(0, len(zpred)):
    if zpred[i] > Zmax:
      zpred[i] = Zmax
    if zpred[i] < Zmin:
      zpred[i] = Zmin

  #Extract position information for eaach point
  ztrain_pos = ztrain_positions.values#Position arrays
  x_train = ztrain_pos[:, -2]
  y_train = ztrain_pos[:, -1]  
  ztest_pos = ztest_positions.values
  x_test = ztest_pos[:, -2]
  y_test = ztest_pos[:, -1]
  
  xz = np.concatenate((x_train, x_test), axis = 0).flatten()
  yz = np.concatenate((y_train, y_test), axis = 0).flatten()
  
  #Convert (x, y) into relative radius
  rz = np.sqrt(xz*xz + yz*yz)/reff
  

  #Running the model on age ********************************************************************
  adataset = smaller.copy()
  adataset = adataset.dropna()
  adataset = adataset.drop(columns = 'ZHL')
  
  atrain_set2 = adataset.sample(frac = 0.5, random_state = 0) #Split inot 2 arrays
  atest_set2 = adataset.drop(atrain_set2.index)
  
  atrain_positions = atrain_set2.copy() #Preserve arrays
  atest_positions = atest_set2.copy()

  atrain_set2 = atrain_set2.drop(columns = 'xkpc')
  atrain_set2 = atrain_set2.drop(columns = 'ykpc')
  atest_set2 = atest_set2.drop(columns = 'xkpc')
  atest_set2 = atest_set2.drop(columns = 'ykpc')

  #True values for age
  atestlabel2 = atest_set2.pop('age')
  atrainlabel2 = atrain_set2.pop('age')

  #Normalisation
  anormtest = norm(atest_set2)
  anormtrain = norm(atrain_set2)
  
  #Prediction
  apredtest = Agemodel.predict(anormtest)
  apredtrain = Agemodel.predict(anormtrain)

  #Rejoining arrays
  apred = np.concatenate((apredtrain, apredtest), axis = 0))#Getting predicted values for Z
  apred = apred.flatten()
  age = np.concatenate((atrainlabel2, atestlabel2), axis = 0).flatten()#flux[:, 1] #True values for Z

  #Boundary values for age
  Amax = 10.2263498
  Amin = 7.84684992
  for i in range (0, len(apred)):
    if apred[i] > Amax:
      apred[i] = Amax
    if apred[i] < Amin:
      apred[i] = Amin
      
  #Position for age values   
  atrain_pos = atrain_positions.values#Position arrays
  x_train = atrain_pos[:, -2]
  y_train = atrain_pos[:, -1]  
  atest_pos = atest_positions.values#Position arrays
  x_test = atest_pos[:, -2]
  y_test = atest_pos[:, -1]
  
  #Coordinate conversion
  xa = np.concatenate((x_train, x_test), axis = 0).flatten()
  ya = np.concatenate((y_train, y_test), axis = 0).flatten()
  
  ra = np.sqrt(xa*xa + ya*ya)/reff[0]
  
  
  #Plotting results************************************************************


  #Values for a summary plot
  for i in range(0, len(zpred)):
    dzpred.append(zpred[i])
    dztrue.append(Z[i])
    dapred.append(apred[i])
    datrue.append(age[i])
    
  #Plot of radius vs {pred, spec} {age, Z}
  plt.clf()
  plt.subplot(221)
  a0, b0, c0 = MCFit(rz, Z)
  plt.title('True Z')
  plt.subplot(222)
  a1, b1, c1 = MCFit(rz, zpred)
  plt.title('Predicted Z')
  plt.subplot(223)
  a2, b2, c2 = MCFit(ra, age)
  plt.title('True Age')
  plt.subplot(224)
  a3, b3, c3 = MCFit(ra, apred)
  plt.title('Predicted Age')
  fname = 'NotBroken/' + phi + 'GalGradients.png'
  plt.legend()
  plt.savefig(fname)
  
  #Plot of error in gradient determination for Z, age
  plt.clf()
  gaussx = np.linspace(min(b0-3*c0, b1 - 3*c1), max(b0+c0*3, b1+3*c1), 1000)
  y0 = gaussian(gaussx, b0, c0)
  y1 = gaussian(gaussx, b1, c1)
  plt.plot(gaussx, y0, 'k',label = 'True Z')
  plt.plot(gaussx, y1, 'r', label = 'Predicted Z')
  plt.xlabel('Gradient')
  fname = 'NotBroken/' + phi + 'ZCurves.png'
  plt.legend()
  plt.savefig(fname)
  
  plt.clf()
  gaussx = np.linspace(min(b2-3*c2, b3 - 3*c3), max(b2+c2*3, b3+3*c3), 1000)
  y2 = gaussian(gaussx, b2, c2)
  y3 = gaussian(gaussx, b3, c3)
  plt.plot(gaussx, y2, 'k',label = 'True age')
  plt.plot(gaussx, y3, 'r', label = 'Predicted age')
  plt.xlabel('Gradient')
  fname = 'NotBroken/' + phi + 'AgeCurves.png'
  plt.legend()
  plt.savefig(fname)
  
  #Values for a summary plot
  dz = a1 - a0
  da = a3 - a2
  dzgrad.append(dz)
  dagrad.append(da)
  dz = b1 - b0
  da = b3 - b2
  MCzgrad.append(dz)
  MCagrad.append(da)
  
  #Error in the gradient difference
  Aerror=np.sqrt(c2**2 + c3**2)
  Zerror = np.sqrt(c0**2 + c1**2)
  
  #Points 1 sigma from gradient difference
  lowage = da - Aerror
  highage = da + Aerror
  lowzed = dz - Zerror
  highzed = dz + Zerror
  lowagrad.append(lowage)
  highagrad.append(highage)
  lowzgrad.append(lowzed)
  highzgrad.append(highzed)


# Creating a summary plot ********************************************************
#Create histogram showing dispersion of predicted Z, age
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

#Best fits to predicted {Z, age} vs spectroscopic {Z, age}
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


muz = np.mean(deltaz)
sigmaz = np.std(deltaz)

sigmaa= np.std(deltaage)
mua = np.mean(deltaage)

#Inclination angle for each galaxy
posheads = ['name', 'n1', 'n2', 'PA', 'ba', 'kpcarc']
positions = pd.read_table('../Galaxy_Parameters.txt', delim_whitespace = True, names = posheads)
Lpos = positions.loc[positions['name'].isin(Lgals)]
Lpos = Lpos.drop(columns = 'name')
temp = np.array(Lpos.values)
inclin = np.array(temp[:, 2])
for i in range(0, len(inclin)):
  inclin[i] = np.float(inclin[i])

#Summary plot of true vs pred Z
deets = figure(plot_width=400, plot_height = 400, title = 'Group L Z, slope = '+ str(slopez))
deets.cross(dztrue, dzpred)
deets.line(xfitz, yfitz, color = 'grey')
deets.yaxis.axis_label = 'Z_pred'
deets.xaxis.axis_label = 'Z_true'
deets.line([-1.2, 0.2], [-1.2, 0.2], color = 'black')

#Dispersion of true vs pred Z
pdfz = max(histz)/(xz* sigmaz * np.sqrt(2*np.pi)) * np.exp(-(np.log(xz)-muz)**2 / (2*sigmaz**2))
hstz = figure(plot_width=400, plot_height = 400, title = 'Histogram for Z, dev = ' + str(sigmaz))
hstz.vbar(x=edgez,  width = 0.1, bottom = 0, top = histz)
hstz.xaxis.axis_label = 'Z_pred - Z_true'
hstz.yaxis.axis_label = 'Count'

#Summary plot of true vs pred age
deeta = figure(plot_width=400, plot_height = 400, title = 'Group L Age, slope = '+str(slopea))
deeta.cross(datrue, dapred)
deeta.line(xfita, yfita, color = 'grey')
deeta.yaxis.axis_label = 'Age_pred'
deeta.xaxis.axis_label = 'Age_true'
deeta.line([8.5, 10.2], [8.5, 10.2], color = 'black')

#Dispersion of true vs pred age
hsta = figure(plot_width=400, plot_height = 400, title = 'Histogram for Age, dev = ' + str(sigmaa))
hsta.vbar(x=edgea,  width = 0.15, bottom = 0, top = hista)
hsta.xaxis.axis_label = 'Age_pred - Age_true'
hsta.yaxis.axis_label = 'Count'


source = ColumnDataSource(data = dict(x1 = MCzgrad, y1 = MCagrad, x2 = dzgrad, y2 = dagrad, lab = Lgals,incl = inclin))
tooltip=[('galaxy', '@lab'),]

#Plots of difference in true/pred age and metallicity gradients
tot = figure(plot_width=400, plot_height = 400, title = 'Group L Gradients', tooltips=tooltip)
tot.circle(0, 0, color = 'black', alpha = 0.5)
tot.cross('x1', 'y1', color = 'red', source = source)
tot.cross('x2', 'y2', source = source)
tot.xaxis.axis_label = 'Z_pred - Z_true'
tot.yaxis.axis_label = 'age_pred - age_true'

#Differnece in gradients (with erros) as a function of galaxy inclination
inc = figure(plot_width=400, plot_height = 400, title = 'Blue = Z, red = age', tooltips=tooltip)
inc.cross('incl', 'x1', source = source)
inc.cross('incl', 'y1',source = source, color = 'red')
inc.segment(y0 = lowzgrad, x0 = inclin, y1 = highzgrad, x1 = inclin)
inc.segment(y0 = lowagrad, x0 = inclin, y1 = highagrad, x1 = inclin, color = 'red')
inc.xaxis.axis_label = 'Inclination'
inc.yaxis.axis_label = 'Predicted - True'

out = gridplot([[deets, hstz], [deeta, hsta], [tot, inc]])
bokeh.io.save(out, 'FullMCUnbroken.html')

plt.clf()

#Plot a scatter plot of gradients with histograms
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

# Plotting the data
ax_scatter.errorbar(MCzgrad, MCagrad,  yerr = ageerror, xerr = zerror, fmt = 'kx', ecolor = 'r')#  the scatter plot
# now determine nice limits by hand:
binwidth = 0.1
lim = np.ceil(np.abs([MCzgrad, MCagrad]).max() / binwidth) * binwidth
ax_scatter.set_xlim((-lim, lim))
ax_scatter.set_ylim((-lim, lim))
ax_scatter.set_xlabel('grad(Z_pred) - grad(Z_spec)')
ax_scatter.set_ylabel('grad(Age_pred) - grad(Age_spec)')
bins = np.arange(-lim, lim + binwidth, binwidth)
ax_histx.hist(MCzgrad, bins=bins, color = 'k') #Top histogram
ax_histx.set_ylabel('Count')
ax_histy.hist(MCagrad, bins=bins, orientation='horizontal', color = 'k') #Right histogram
ax_histy.set_xlabel('Count')

ax_histx.set_xlim(ax_scatter.get_xlim())
ax_histy.set_ylim(ax_scatter.get_ylim())

plt.savefig('ScatteredHists.png')


