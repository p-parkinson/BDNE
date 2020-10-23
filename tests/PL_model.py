# Test code for PL model fitting
# Examples of how to implement the fitter
# Author  : Stephen Church <stephen.church@manchester.ac.uk>
# Version : 0.2

import time

from BDNE.physics.photoluminescence_models import *
from BDNE import data_structures

# Create a Wire collection
w = data_structures.WireCollection()
# Populate from the database, sample ID 25
w.loadsample(25)
# List how many wires samples
print('Wire Collection has {} wires'.format(len(w.db_ids)))

##############################################################
# initialising the fitter
##############################################################

# create a PL fitter with a defined model
fitter = PLfit(pl_3d_boltz)
# REQUIRED: define initial conditions for the model [sigma, E0, T, A]
fitter.par0 = [0.03, 2.38, 300, 1]
# REQUIRED: define bounds for the model [sigma, E0, T, A]
fitter.bounds = [(0.005, 0.1), (2.1, 2.5), (300, 500), (0, 2)]

# Optional:
fitter.clip_spectra = True
fitter.spec_lims = [1.75, 3]
fitter.A_lim = 3
fitter.plot_output = False

###############################################################
# Iterating using the fitter
###############################################################
# Number of spectra to study
n_spec = 300
print('Begin fitting loop')
par = np.zeros((n_spec,len(fitter.par0)))
i, dy_res = 0, 0
# Select a subset of wires for demonstration
sample = w.sample(n_spec)
# Collect spectral data (avoid repeated DB hits)
data = sample.get('spectra').collect()
# Iterate and fit
start = time.time()
for (wire, PL) in zip(sample, data):
    print('{} : wire ID {}'.format(i, wire.db_id))
    par[i, :] = fitter(PL)

    # estimate the time remaining
    i = i + 1
    if not (i % 10):
        elapsed = time.time() - start
        print('Elapsed time: {}s, ETA : {}s ({}s/w)'.format(elapsed, round((n_spec - i - 1) * elapsed / (i + 1)),elapsed/i))

# Cut par down to size, removing blank sets
par = par[0:(i-1), :]

###############################################################
# Show results
###############################################################

labels = ['sigma', 'E_g', 'T', 'A']
plt.clf()
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.hist(par[:, i], round(n_spec/10))
    plt.xlabel(labels[i])
plt.show()
