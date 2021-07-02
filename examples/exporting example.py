# Example usage for BDNE code
# Authors : Patrick Parkinson <patrick.parkinson@manchester.ac.uk>
# and Andras Botar <andras.botar@student.manchester.ac.uk>
import pandas as pd

from BDNE import connect_big_query, connect_mysql
from BDNE.data_structures import *

# Choose connection type
#connect_big_query()
connect_mysql()

# Set up an entity collection
w = EntityCollection()
# Populate from the database from an EntityGroup
w.load_entity_group(3)
# List how many entities the EntityGroup has
print('Wire Collection has {} entities'.format(len(w)))

# Get one entity to test at random
entity = w.sample(1)
# State which one samples
print('Selected entity ID {}'.format(entity.db_id))
# List all the data associated with this wire
wire_data = entity.get_data()
print(wire_data)

# get a batch of entities from the entity collection
batch_data = w.get_batch()
# Save data as a .CSV file
batch_data.to_csv("batch_data.csv")

# Save data as a compressed hdf5 file, in the "fixed" and not the "table" format
batch_data.to_hdf('fixed.h5' ,'table', mode='w', complevel=9, complib='bzip2',format='fixed')

#read the dataframe from disk
read_data = pd.read_hdf('fixed.h5','table')
#check if the data read frokm the file is the same as the original:
if read_data.equals(batch_data):
    print("The orignal and re-read data are the same")
else:
    print("There was a difference caused by storing and re-reading the data")

print("exports completed")