# Definition of data structures for BDNE project
# Author : Patrick Parkinson <patrick.parkinson@manchester.ac.uk>

# For sampling from set
import random

# For conversion
import numpy as np

# For cacheing
from cachetools import cached

# Import for database
from BDNE.db_orm import *
from BDNE import session


#################################################################
#   A single wire class
#################################################################
class Wire:
    """A class for a single nanowire"""
    db_id = None
    experiment_container = []

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
            return stm.all()[0][0]
        else:
            raise KeyError('Measurement ID {} not found in database'.format(exp_id))


#################################################################
#   WireCollection (a collection of wires)
#################################################################

class WireCollection:
    """A collection of wires. Lazy handling, stores only db_ids for the wires and returns either a wire, a set of wires,
     or a set of measurements.
      Typical usage:
      ``w = WireCollection();
      w.loadsample(25);``"""
    db_ids = []
    cursor = -1

    def __repr__(self):
        """Representation"""
        return "{} IDs={}".format(self.__class__.__name__, len(self.db_ids))

    def __init__(self, startid=None):
        # Set up wire collection, either blank or with a set of entity IDs.
        if type(startid) is list:
            self.db_ids = startid

    def load_sample(self, sampleid):
        stm = session.query(Entity.ID).filter(Entity.sampleID == sampleid)
        self.db_ids = [i[0] for i in stm.all()]
        if not self.db_ids:
            raise Warning('No wires found with sample ID {}'.format(sampleid))

    def load_entitygroup(self, entityGroupid):
        # TODO: implement entitygroup loading
        stm = session.query(EntityGroup_Entity.entityID).filter(EntityGroup.ID == entityGroupid)
        self.db_ids = [i[0] for i in stm.all()]
        if not self.db_ids:
            raise Warning('No wires found with sample ID {}'.format(entityGroupid))

    def sample(self, k=1):
        # Get a random subset
        if type(k) == int:
            wid = random.choices(self.db_ids, k=k)
            if len(wid) == 1:
                return Wire(wid[0])
            else:
                return WireCollection(wid)
        elif type(k) == list:
            return WireCollection(self.db_ids[k])
        else:
            raise TypeError('Argument to sample must be either an integer or a list of integers.')

    def mask(self, idset):
        """Create a set from the intersection with other ids"""
        if type(idset) is WireCollection:
            idset = idset.db_ids
        elif type(idset) is list:
            pass
        else:
            raise TypeError('Mask must be passed either a list of entity IDs or another WireCollection')
        # Create an intersection (where
        intersection = set(self.db_ids).intersection(idset)
        return WireCollection(list(intersection))

    def get(self, wid):
        # Two approaches
        if type(wid) is int:
            # Return single wire
            return Wire(self.db_ids[wid])
        if type(wid) is str:
            # Return a measurement collection with associated entity (to backreference)
            stm = session.query(Measurement.ID, Entity.ID).select_from(Measurement).join(Object).join(Entity).join(
                Experiment).filter(
                Entity.ID.in_(self.db_ids), Experiment.type == wid)
            ret = stm.all()
            return MeasurementCollection([i[0] for i in ret], [i[1] for i in ret])

    def __next__(self):
        # To iterate
        self.cursor = self.cursor + 1
        if self.cursor == len(self.db_ids):
            self.cursor = 0
            raise StopIteration()
        return self.get(self.cursor)

    def __iter__(self):
        return self


#################################################################
#   MeasurementCollection
#  TODO: create as_pandas() to collect as a pandas list
#  TODO: Create "clever" memoization, for each id.
#################################################################

class MeasurementCollection:
    db_ids = []
    entity_ids = []
    cursor = -1
    post_process = None

    def __init__(self, measurementids=None, entityids=None):
        if type(measurementids) is list:
            if len(measurementids) == len(entityids):
                self.db_ids = measurementids
                self.entity_ids = entityids
            else:
                raise RuntimeError('Both measurementid and entityid must be provided with the same length.')

    def __repr__(self):
        """Representation"""
        return "{} IDs={}".format(self.__class__.__name__, len(self.db_ids))

    def sample(self, k=1):
        # Get a random selection of measurements
        selected = random.choices(range(len(self.db_ids)), k=k)
        stm = self._get(selected)

    @cached(cache={})
    def _get(self, n):
        # Convert ranges
        if type(n) is range:
            n = list(n)
        # Check if a list passed (must be)
        if type(n) is not list:
            raise NotImplementedError('n must be a list')
        # Convert to numpy
        n = np.array(n)
        # Range check
        if np.any(n > len(self.db_ids)) or np.any(n < 0):
            raise KeyError('Index must be in range 0 to {}'.format(len(self.db_ids)))
        # Convert indices to db_ids
        to_get = [self.db_ids[i] for i in n]
        # Three methods - single, zero length, multiple
        if len(to_get) == 1:
            stm = session.query(Measurement.data).filter(Measurement.ID == to_get[0])
            to_return = np.array(stm.first()[0]).squeeze()
        elif len(to_get) == 0:
            to_return = None
        else:
            stm = session.query(Measurement.data).filter(Measurement.ID.in_(to_get))
            to_return = [np.array(i[0]).squeeze() for i in stm.all()]
        return to_return

    def collect(self):
        # Get all measurements
        to_ret = self._get(range(len(self.db_ids)))
        if self.post_process:
            # A post process function exists, apply
            return np.array([self.post_process(i) for i in to_ret])
        else:
            return to_ret

    def mask(self, idset):
        """Create a set from the intersection with other ids"""
        if type(idset) is MeasurementCollection:
            idset = idset.entity_ids
        elif type(idset) is list:
            pass
        else:
            raise TypeError('Mask must be passed either a list of entity IDs or another MeasurementCollection')
        # Create an intersection on entity IDs
        (intersection, id1, id2) = np.intersect1d(self.entity_ids, idset, return_indices=True)
        # Filter measurement IDs
        measurement_ids = [self.db_ids[i] for i in id1]
        # Return a new filtered measurement collection
        return MeasurementCollection(measurementids=measurement_ids, entityids=intersection)

    def __next__(self):
        # To iterate
        self.cursor = self.cursor + 1
        if self.cursor == len(self.db_ids):
            self.cursor = 0
            raise StopIteration()
        toret = self._get([self.cursor])
        if self.post_process:
            return np.array(self.post_process(toret))
        else:
            return toret

    def __iter__(self):
        return self


#################################################################
#   PostProcess
#################################################################

class PostProcess:
    mc = None
    func = None
    _cursor = 0

    def __init__(self, mc=None):
        if type(mc) is MeasurementCollection:
            self.mc = mc

    def __repr__(self):
        """Representation"""
        if self.func:
            fname = self.func.__name__
        else:
            fname = 'None'
        if self.mc:
            mcname = repr(self.mc)
        else:
            mcname = 'None'
        return "Postprocessing function [{}] attached to {}".format(fname, mcname)

    def set_function(self, func):
        self.func = func

    def set_data(self, mc):
        self.mc = mc

    def __next__(self):
        # To iterate
        self._cursor = self._cursor + 1
        if self._cursor == len(self.mc.db_ids):
            self._cursor = 0
            raise StopIteration()
        toret = self.mc._get([self._cursor])
        return np.array(self.func(toret))

    def __iter__(self):
        return self

    def collect(self):
        # Get all measurements
        to_ret = self.mc._get(range(len(self.mc.db_ids)))
        return np.array([self.func(i) for i in to_ret])