# Test code for BDNE
# Author : Patrick Parkinson <patrick.parkinson@manchester.ac.uk>

import matplotlib.pyplot as plt

import data_structures

####################################################################
# Set up a wire collection
w = data_structures.WireCollection()
# Populate from the database, sample ID 25
w.loadsample(25)
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
PL = w.get('spectra')
# State how many obtained
print('Obtained measurement set with {} measurements'.format(len(PL.db_ids)))
# Add a post-process sum onto the spectra
PL.post_process = sum
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
plt.hist(PL.collect(),range=(-50000,50000), bins=40)
plt.xlabel('Intensity (um)')
plt.ylabel('Frequency')
plt.subplot(224)
plt.hist(PL.collect(),100)
plt.title('PL intensity of {} wires'.format(len(w.db_ids)))
plt.xlabel('Intensity (um)')
plt.ylabel('Frequency')

# Update plot
plt.show()