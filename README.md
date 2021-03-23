# Big-data for Nano-Electronics - BDNE #

**Author** - Patrick Parkinson [patrick.parkinson@manchester.ac.uk](mailto:patrick.parkinson@manchester.ac.uk)

**License** - This software is licensed under the [MIT-license](https://opensource.org/licenses/MIT)

## Purpose ##
Datasets associated with nano-electronics and nano-opto-electronics can be highly 
heterogeneous and large scale. This package provides tools to work with a specific
back-end and manage point, spectral and imaging data.

## Design ##
BDNE is code to connect to and explore the **big-data for nano-electronics**
dataset. It allows retrieval of _measurements_ (results of _experiments_) for
_objects_ which are associated with unique _entities_ for different _samples_. Each
term is linked to a database table.

The database is designed to hold and store information for functional nanomaterials,
and to allow analysis of this data. There is no specific schema for storage of data, which 
includes point measurement (such as length or width), one-dimensional measurements (such as spectra) 
two-dimensional measurements (including images) and structured results (for functional data).

## How do I get set up? ##

The code is designed to run on Anaconda, but has been demonstrated to work on 
[Google Colab](https://colab.research.google.com/) with minimal effort.

* Set up - *Python 3* is recommended.
* Configuration file
* Dependencies - *numpy, sqlalchemy, pandas, zlib, scipy, pyarrow, pybigquery*
* Database configuration - runs against BDNE database server via mysql or through the public repository on Google BigQuery.

## How do I use it? ##
### Configuration ###
To connect to the public BigQuery datasource, a credentials file is required. Contact Patrick Parkinson 
[patrick.parkinson@manchester.ac.uk](mailto:patrick.parkinson@manchester.ac.uk) for details.

### High-level functionality ###

Four classes are implemented for higher-level functionality, within the ```data_structures.py``` file.
* ```Wire``` - A container class for a single entity, and all associated measurements.
* ```WireCollection``` - A container class for a collection of entities.
* ```MeasurementCollection``` - A container class for a collection of measurements.
* ```PostProcess``` - A class to wrap MeasurementCollection or PostProcess collections with a processing function. 

These should be used for handling data and processing, and connect into the ```db_orm``` backend.

### Example code ###

Basic functionality can be produced using the example code provided in ```BDNE_example_code.py```. 
A typical workflow might be:

```python

from BDNE import connect_big_query
from BDNE import data_structures

# Connect to the database back-end
connect_big_query()

# Create an empty container
wc = data_structures.WireCollection()
# Load data related to sample 25
wc.load_sample(25)

# It is possible to iterate over each wire, 
# and retrieve a single measurement using the iterator approach
for wire in wc:
    # Process this wire, i.e. spectra
    s = wire.get("spectra")

# However, it is more efficient to collect all measurements into
# a Measurement Collection.
data = wc.get_measurement("l")

# You can iterate over the data, but this tends to be slow for large datasets
# as it requires many database calls

def process(da):
    # Dummy processing function
    return da

for d in data:
    # Iterate over every l value
    process(d)
    
# It is preferred to collect all of the data from the server, and then to process.
# This can lead to potentially large datasets of GB for imaging (single SQL call, get all data at once)
d = data.collect()

# The preferred method is to wrap the data with a PostProcess
p = data_structures.PostProcess(data)
# Set the processing function
p.set_function(process)
# Collect or iterate:
processed_data = p.collect()

```

## Contributors

This code is the product of input from researchers and students at the University of Manchester, with specific input from:
* **Stephen Church** [stephen.church@manchester.ac.uk](mailto:stephen.church@manchester.ac.uk) - Photoluminescence fitting
* **Rafe Whitehead, Thomas Blackmore, Jonathan Ryding** - Masters students contributing to testing.

The underlying dataset contains materials created via contributions from researchers around the world.