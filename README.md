# README - BDNE #

**Author** - Patrick Parkinson [patrick.parkinson@manchester.ac.uk](mailto:patrick.parkinson@manchester.ac.uk)

BDNE is code to connect to and explore the big-data for nano-electronics
dataset. It allows retrieval of _measurements_ (results of _experiments_) for
_objects_ which are associated with unique _entities_ for different _samples_. Each
term is linked to a database table.

The database is designed to hold and store information for functional nanomaterials,
and to allow analysis of this data.

### How do I get set up? ###

* Set up - *Python 3*
* Configuration
* Dependencies - *matplotlib, numpy, sqlalchemy, pandas, zlib, scipy*
* Database configuration - runs against BDNE database server such as *db_uom.oms-lab.org*

### How do I use it? ###
#### Low level functionality ####
You must have a copy of config.ini which should look like:
```
[DATABASE]
user = ###
pass = ###
server = db_uom.oms-lab.org
port = 3306
```
You can import the main function using `import db_orm`. 

#### High-level functionality ####

Three classes are implemented for higher-level functionality, within the ```data_structures``` file.
```
Wire
WireCollection
MeasurementCollection
```
These should be used for handling data and processing, and connect into the ```db_orm``` backend.

Example code is provided in ```main.py```. The containers WireCollection and MeasurementCollection are iterable:
```python
import data_structures
wc = data_structures.WireCollection()
wc.loadsample(25)
for wire in wc:
    # Process this wire, i.e. spectra
    s = wire.get('spectra')

data = wc.get('l')
# SLOW, inefficient as results in a lot of server overhead
for d in data:
    # Iterate over every l value
    process(d)
# FASTER but slow setup potentially large memory requirements 
# (single SQL call, get all data at once)
d = data.collect()
```
