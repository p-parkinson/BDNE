# BDNE package [https://bitbucket.org/paparkinson1/bdne_db]
# Author : Patrick Parkinson <patrick.parkinson@manchester.ac.uk>

# Configparser used for configuration settings
import configparser
# OS used for configuration settings
import pathlib
# Connect to database
from BDNE.db_orm import connect
import BDNE.config as cfg


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
def connect_mysql(server: str = "", port: str = "", user: str = "", passwd: str = "") :
    if len(server) == 0:
        server = config.get('DATABASE', 'server')
    if len(port) == 0:
        port = config.get('DATABASE', 'port')
    if len(user) == 0:
        user = config.get('DATABASE', 'user')
    if len(passwd) == 0:
        passwd = config.get('DATABASE', 'pass')
    cfg.session = connect(mysql={'server': server, 'port': port,
                             'user': user, 'password': passwd})


def connect_big_query(credentials_json: str = ""):
    if len(credentials_json) == 0:
        credentials_json = package_directory / 'bdne_bigquery.json'
    cfg.session = connect(big_query={'bigquery_uri': 'bigquery://bdne-307813/primary',
                                     'credentials_path': credentials_json})

#connect_big_query()
