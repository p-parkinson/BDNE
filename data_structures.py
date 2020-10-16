# Definition of data structures for BDNE project
# Author : Patrick Parkinson <patrick.parkinson@manchester.ac.uk>
# Date : 14/10/2020

# Import for database
from db_orm import *
import random


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

    def __init__(self, db_id=None):
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
#################################################################

class WireCollection:
    db_ids = []

    def __repr__(self):
        """Representation"""
        return "{} IDs={}".format(self.__class__.__name__, len(self.db_ids))

    def __init__(self, startid=None):
        if type(startid) is list:
            self.db_ids = startid

    def loadsample(self, sampleid):
        stm = session.query(Entity.ID).filter(Entity.sampleID == sampleid)
        self.db_ids = [i[0] for i in stm.all()]
        if not self.db_ids:
            raise Warning('No wires found with sample ID {}'.format(sampleid))

    def get(self, ID):
        # Two approaches
        if type(ID) is int:
            # Return single wire
            return Wire(self.db_ids[ID])
        if type(ID) is str:
            # Return a measurement collection
            stm = session.query(Measurement.ID).join(Object).join(Entity).join(Experiment).filter(
                Entity.ID.in_(self.db_ids), Experiment.type == ID)
            return MeasurementCollection([i[0] for i in stm.all()])


#################################################################
#   MeasurementCollection
#################################################################

class MeasurementCollection:
    db_ids = []
    cursor = -1

    def __init__(self, measurementids=None):
        if type(measurementids) is list:
            self.db_ids = measurementids

    def __repr__(self):
        """Representation"""
        return "{} IDs={}".format(self.__class__.__name__, len(self.db_ids))

    def sample(self, k=1):
        # Get a random selection of measurements
        selected = random.choices(self.db_ids, k=k)
        stm = session.query(Measurement.data).filter(Measurement.ID.in_(selected))
        return [i[0] for i in stm.all()]

    def collect(self):
        # Get all measurements
        stm = session.query(Measurement.data).filter(Measurement.ID.in_(self.db_ids))
        return [i[0] for i in stm.all()]

    def __next__(self):
        # To iterate
        self.cursor = self.cursor + 1
        stm = session.query(Measurement.data).filter(Measurement.ID == self.db_ids[self.cursor])
        return stm.first()

    def __iter__(self):
        return self
