# BDNE package [https://bitbucket.org/paparkinson1/bdne_db]
# Author : Patrick Parkinson <patrick.parkinson@manchester.ac.uk>

# Configparser used for configuration settings
import configparser
# OS used for configuration settings
import pathlib
# Connect to database
from BDNE.db_orm import connect

#####################################################
# Get database configuration
#####################################################
# Get current path (required for locating config.ini)
package_directory = pathlib.Path(__file__).parent.absolute()
# Read and import configuration from config file
config = configparser.ConfigParser()
config.read(package_directory / 'config.ini')


#####################################################
# Connect to database
#####################################################
def reconnect():
    return connect(config.get('DATABASE', 'server'),
                   config.get('DATABASE', 'port'),
                   config.get('DATABASE', 'user'),
                   config.get('DATABASE', 'pass'))


session = reconnect()
