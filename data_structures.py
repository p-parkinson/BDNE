# Definition of data structures for BDNE project
# Author : Patrick Parkinson <patrick.parkinson@manchester.ac.uk>
# Date : 14/10/2020

# Import for database
from db_orm import *


#################################################################
#   A single wire class
#################################################################
class Wire:
    """A class for a single nanowire"""
    db_id = None
    experiment_container = []
    cache = False

    def __repr__(self):
        """Representation"""
        return "{} ID={}".format(self.__class__.__name__, self.db_id)

    def __init__(self, db_id=None, cache=False):
        """Initialise the wire class as empty or with a db_id. Cache is optional, can reduce database hits."""
        if db_id is None:
            return
        # ID given
        self.db_id = db_id

    def populate_from_db(self):
        """Retrieve all experiments associated with this nanowire ID"""
        stm = session.query(Experiment.type, Measurement.ID).join(Measurement).join(Object).join(Entity).filter(
            Entity.ID == self.db_id)
        if not stm.all():
            raise KeyError('No Entity exists with ID {}'.format(self.db_id))
        self.experiment_container = stm.all()

    def experiments(self):
        """List all experiments in this wire"""
        if not self.experiment_container:
            self.populate_from_db()
        return [i[0] for i in self.experiment_container]

    def get(self, experiment):
        """Get a single experiment associated with this wire by experiment number or name"""
        # Check if we have downloaded experiment list yet
        if not self.experiment_container:
            self.populate_from_db()
        # Check type of experiment
        if type(experiment) is int:
            exp_id = self.experiment_container[experiment][1]
        elif type(experiment) is str:
            exp_id = [i[1] for i in self.experiment_container if i[0] == experiment]
            if len(exp_id) == 0:
                raise KeyError('Experiment {} not present for wire ID {}'.format(experiment, self.db_id))
            elif len(exp_id) == 1:
                exp_id = exp_id[0]
            else:
                raise KeyError('Experiment {} ambiguous for wire ID {}'.format(experiment, self.db_id))
        else:
            raise TypeError('Experiment must be defined as an integer or a string')
        # Retrieve experiment results from database
        stm = session.query(Measurement.data).filter(Measurement.ID == exp_id)
        if len(stm.all()) == 1:
            return stm.all()[0]
        else:
            raise KeyError('Measurement ID {} not found in database'.format(exp_id))


#################################################################
#   WireCollection (a collection of wires)
#   TODO: Add methods for single wire, group of wire, and data retrieval
#################################################################

class WireCollection:
    db_ids = []

    def __repr__(self):
        """Representation"""
        return "{} IDs={}".format(self.__class__.__name__, len(self.db_ids))


#################################################################
#   MeasurementCollection
#################################################################

class MeasurementCollection:
    db_ids = []

    def __repr__(self):
        """Representation"""
        return "{} IDs={}".format(self.__class__.__name__, len(self.db_ids))
