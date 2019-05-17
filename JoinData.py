import numpy as np


LongList = open('LongList.txt', 'r') #List of all galaxies in this sample
output = open('Groups/Alldata.dat', 'w')
#Output of the file will be:
#Age (L)       Z (L)     Colour....
for item in LongList: #Iterate over each galaxy
  print(item)
  name = item.split()
  filename = 'JPAS_SIM/' + name[0] + 'jpas_SP.dat'  #File containing galaxy's metallicity, age, fluxes  
  posfile = 'Positions/Corrected/' + name[0] + '_corrected.dat' #Positional information for each spaxel
  data = np.loadtxt(filename)
  posdata = np.loadtxt(posfile)
  appendable = data[:, 3:] #Relevant data
  positions = posdata[:, 2:]
  j = 0 #Parameter to iterate through each point in position
  for row in appendable: #Iterate through gspaxels for the galaxy
    output.write(name[0]+ '  ')
    for i in range(0, len(row)):
      if i == 0 or i == 1: # age and metallicity values, which should not be changed
	output.write(str(row[i])+'   ')
      elif i == 2:
	ref = row[i] #Reference flux
      else:
	val = row[i] - ref #Conversion to colour
	output.write(str(val)+'   ')

    output.write(str(positions[j, 0])+'   ') #Position of each spaxel
    output.write(str(positions[j, 1])+'   ')
    j = j + 1
    output.write('\n')
  
output.close()
LongList.close()

