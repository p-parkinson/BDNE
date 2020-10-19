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
lengths = w.get('l')
# State how many obtained
print('Obtained measurement set with {} measurements'.format(len(lengths.db_ids)))

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
plt.subplot(212)
plt.hist(lengths.collect())
yl = plt.ylim()
plt.plot(wire.get('l')*[1,1],yl)
plt.title('Lengths of {} wires'.format(len(w.db_ids)))
plt.xlabel('Length (um)')
plt.ylabel('Frequency')

# Update plot
plt.show()