# Example usage for BDNE
# Author : Patrick Parkinson <patrick.parkinson@manchester.ac.uk>

import matplotlib.pyplot as plt
import numpy as np
from BDNE import connect_big_query, connect_mysql
from BDNE.data_structures import *

# Choose connection type
#connect_big_query()
connect_mysql()

####################################################################
# Set up a wire collection
w = WireCollection()
# Populate from the database, EntityGroup 3
w.load_entity_group(4)
# List how many wires samples
print('Wire Collection has {} wires'.format(len(w.db_ids)))

#############
# Get one wire to test at random
wire = w.sample(1)
# State which one samples
print('Selected wire ID {}'.format(wire.db_id))
# List what data this wire has
print(wire.experiments())

##############
# Get all wire lengths
spectra = w.get_measurement('spectra')
# State how many obtained
print('Obtained measurement set with {} measurements'.format(len(spectra)))
# Add a post-process sum onto the spectra
PL = PostProcess(spectra)
pkpos = PostProcess(spectra)
PL.set_function(np.sum)
pkpos.set_function(np.argmax)
####################################################################
# Plot the image and spectra
plt.clf()

# Show image
plt.subplot(221)
plt.imshow(wire.get('pic'))
plt.title('Pic {}'.format(wire.db_id))

# Show spectra
plt.subplot(222)
plt.plot(wire.get('spectra'))
plt.title('Spectra')
plt.ylabel('PL (arb)')
plt.xlabel('Spectra (arb)')

# Show histogram of length data
plt.subplot(223)
plt.hist(pkpos.collect()['processed'], 100)
plt.xlabel('PL peak position')
plt.ylabel('Frequency')
plt.subplot(224)
plt.hist(PL.collect()['processed'], 100)
plt.title('PL intensity of {} wires'.format(len(w.db_ids)))
plt.xlabel('Intensity (um)')
plt.ylabel('Frequency')

# Update plot
plt.show()
