# SQLAlchemy-based ORM model for the BDNE project
# Author : Patrick Parkinson <patrick.parkinson@manchester.ac.uk>
# Date : 30/06/2020

from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import configparser
import os

# Get current path (required for locating config.ini)
package_directory = os.path.dirname(os.path.abspath(__file__))

# Read and import configuration from config file
config = configparser.ConfigParser()
config.read(os.path.join(package_directory, 'config.ini'))

# Create base autoloader
Base = automap_base()

# Connect to the database using credentials from config.ini
engine = create_engine('mysql+mysqlconnector://%s:%s@db.oms-lab.org:3306/bdne' % (
    config.get('DATABASE', 'user'), config.get('DATABASE', 'pass')))

# reflect the tables
Base.prepare(engine, reflect=True)

# Set up the tables
Measurement = Base.classes.measurement
Object = Base.classes.object
Entity = Base.classes.entity
Sample = Base.classes.sample
Experiment = Base.classes.experiment
EntityGroup = Base.classes.entityGroup
# Set up metadata engine
Metadata = Base.classes.metadata
# Set up a session
session = Session(engine)
