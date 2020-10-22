#Test code for PL model fitting
#Examples of how to implement the fitter
# Author  : Stephen Church <stephen.church@manchester.ac.uk>
# Version : 0.1

from photoluminescence_models import *
import data_structures
import matplotlib.pyplot as plt
import time

w = data_structures.WireCollection()
# Populate from the database, sample ID 25
w.loadsample(25)
# List how many wires samples
print('Wire Collection has {} wires'.format(len(w.db_ids)))

# Get one wire to test at random
wire = w.sample(1)
# State which one sampled
print('Selected wire ID {}'.format(wire.db_id))

# Get spectrum
PL = wire.get('spectra')
#plt.plot(PL)

##############################################################

#initialising the fitter

#create a PL fitter with a defined model
fitter = PLfit('PL_3D_Boltz')
#REQUIRED: define initial conditions for the model [sigma, E0, T, A]
fitter.par0 = [0.05,1.5,300,1]
#RERUIRED: define bounds for the model [sigma, E0, T, A]
fitter.bounds = [(0.01,0.1),(1,3),(300,300),(0,None)]


####################################

#OPTIONAL parameters to initialise:
    
#import spectral response
fitter.spectral_correct = 0
fitter.response = np.array(1044*[1])

#import x axis (energy)
wavelength = []
for i in range(1044):
    wavelength.append(346 + 0.796*i - 3.79e-5*i**2)
fitter.eV = 1239.842/np.array(wavelength)

#clip the spectra
fitter.clip_spectra = 0
fitter.spec_lims = [1.3,1.7]

#plot the results
fitter.plot_output = 0

#set width and amplitude thresholds to remove cosmic rays and bad spectra
fitter.width_thresh = 10
fitter.A_lim = 3


##################################################

#single use of the fitter

#fit the spectrum
fitter(PL)

#fitting parameters
print(fitter.output_par)
print(fitter.output_res)

#fit spectrum
#print(fitter.output_PL)

###############################################################

#Iterating using the fitter
n_spec = 300
start = time.time()
print('Begin fitting loop')
par = []
dy_res = 0
for i in range(n_spec):
    wire = w.sample(1)
    print('Selected wire ID {}'.format(wire.db_id))
    PL = wire.get('spectra')
    fitter(PL)

    output = [wire.db_id,fitter.output_par,fitter.output_res]
    par.append(output)
    
    #use dynamic initial conditions, recommended to speed up loop
    #initial parameter is average of fit results
    dy_res += fitter.output_par
    fitter.par0 = dy_res/(i+1)
    
    #estimate the time remaining
    if i % 10 == 0:
        elapsed = time.time()
        print('Elapsed time: ' + str(round(elapsed-start)) + 's, estimated time remaining: ' + str(round((n_spec-i-1)*(elapsed-start)/(i+1))) + ' s')
    
del dy_res, output, elapsed, start