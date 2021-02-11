# Definition of data structures for BDNE project
# Author : Patrick Parkinson <patrick.parkinson@manchester.ac.uk>

# For sampling from set
import random

# For conversion
import numpy as np

# For datasets
import pandas as pd

# Import for database
from BDNE.db_orm import *
from BDNE import session


#################################################################
#   A cache class for storing data locally
#################################################################

class DBCache:
    """A simple cache class - never kicks out old data unless told to"""
    _cache: pd.DataFrame

    def __init__(self):
        """Set up pandas dataframe to store data"""
        self._cache = pd.DataFrame()

    def clear(self):
        """Empty the cache"""
        self._cache = pd.DataFrame()

    def __call__(self, ids):
        """Convenience function"""
        return self.check(ids)

    def check(self, ids):
        """Look for hits with ids, must be unique"""
        if len(ids) == 0:
            return None, None
        ids = np.array(ids)
        # Get from cache
        cached = self._cache[self._cache.index.isin(ids)]
        # List unfound items to be read in
        unfound = np.setdiff1d(ids, cached.index.to_numpy()).tolist()
        return unfound, cached

    def update(self, ids, data):
        # Convert to integer
        ids = ids.astype('int')
        # Make sure not to update existing data
        missing = np.setdiff1d(ids, self._cache.index.to_numpy())
        # update index
        oldidx = data.index
        data.index = ids
        # Create new dataframe
        self._cache = self._cache.append(data[data.index.isin(missing)])
        # Restore
        data.index = oldidx

    def __len__(self):
        return self._cache.memory_usage(deep=True)


#################################################################
#   A single wire class
#################################################################
class Wire:
    """All data for a unique single element."""
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
    """A collection of elements.
    Lazy handling, stores only db_ids for the wires and returns either a wire, a set of wires, or a set of measurements.
      Typical usage:
      ``w = WireCollection();
      w.load_sample(25);``"""

    db_ids = []
    cursor = -1

    def __repr__(self):
        """Representation"""
        return "{} IDs={}".format(self.__class__.__name__, len(self.db_ids))

    def __init__(self, start_id=None):
        """Set up wire collection, either blank or with a set of initial entity IDs."""
        if type(start_id) is list:
            self.db_ids = start_id

    def __len__(self):
        """The number of wires in this collection"""
        return len(self.db_ids)

    def load_sample(self, sample_id: int):
        """Load a sample ID into the WireCollection class"""
        stm = session.query(Entity.ID).filter(Entity.sampleID == sample_id)
        self.db_ids = [i[0] for i in stm.all()]
        if not self.db_ids:
            raise Warning('No wires found with sample ID {}'.format(sample_id))

    def load_entity_group(self, entity_group_id: int):
        """Load an entityGroup ID into the WireCollection class"""
        # TODO: implement entity_group loading
        stm = session.query(EntityGroupEntity.entityID).filter(EntityGroup.ID == entity_group_id)
        self.db_ids = [i[0] for i in stm.all()]
        if not self.db_ids:
            raise Warning('No wires found with sample ID {}'.format(entity_group_id))

    def sample(self, k=1):
        """Return a random subset of k entities from the WireCollection."""
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

    def mask(self, id_set):
        """Create a new entity set from an intersection with other entity ids"""
        if type(id_set) is WireCollection:
            id_set = id_set.db_ids
        if type(id_set) is MeasurementCollection:
            id_set = id_set.entity_ids
        elif type(id_set) is list:
            pass
        else:
            raise TypeError('Mask must be passed either a list of entity IDs or another WireCollection')
        # Create an intersection (where
        intersection = set(self.db_ids).intersection(id_set)
        return WireCollection(list(intersection))

    def logical_mask(self, mask):
        """Create a new wire collection using a logical mask"""
        new_ids = np.array(self.db_ids)[np.array(mask)].tolist()
        return WireCollection(new_ids)

    def get(self, wid):
        """Return either a single entity (when enumerated) or a MeasurementCollection (when a string is passed)"""
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
        """To iterate over each wire in the Collection"""
        self.cursor = self.cursor + 1
        if self.cursor == len(self.db_ids):
            self.cursor = 0
            raise StopIteration()
        return self.get(self.cursor)

    def __iter__(self):
        return self

    def __add__(self, other):
        """Combine two entityCollections and return a merged set"""
        return WireCollection(self.db_ids + other.db_ids)


#################################################################
#   MeasurementCollection
#  #  TODO: Create "clever" memoization, for each id.
#################################################################

class MeasurementCollection:
    """A class to hold a collection of related measurement.
    Uses lazy loading, holding only the database IDs and associated entity IDs until a get() or collect() is issued."""
    db_ids = []
    entity_ids = []
    cursor = -1
    _db_cache = DBCache()
    _use_cache = True

    def __init__(self, measurement_ids=None, entity_ids=None):
        """Initialise with a list of measurement_IDs and entity_ids"""
        # TODO: a list is a bad type here, as it does not guarantee preserving order. Use numpy.array
        if type(measurement_ids) is list:
            if len(measurement_ids) == len(entity_ids):
                self.db_ids = measurement_ids
                self.entity_ids = entity_ids
            else:
                raise RuntimeError('Both measurement_id and entity_id must be provided with the same length.')

    def __repr__(self):
        """Representation"""
        return "{} IDs={}".format(self.__class__.__name__, len(self.db_ids))

    def __len__(self):
        """Return the number of measurements in this collection"""
        return len(self.db_ids)

    def sample(self, k=1):
        """Get a random selection of k measurements"""
        selected = random.choices(range(len(self.db_ids)), k=k)
        return self._get(selected)

    def _get(self, n):
        """A cached function to return measurements from the set."""
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
        # Zero length, multiple
        if len(to_get) == 0:
            # Nothing to return
            return None
        else:
            # Multiple datasets to return
            # Need to check cache
            if self._use_cache:
                (to_get, cached) = self._db_cache.check(to_get)
            else:
                cached = None
            # Collect the rest
            stm = session.query(Measurement.data, Object.entity_id, Measurement.ID).join(Object).filter(
                Measurement.ID.in_(to_get))
            stmall = stm.all()
            # Format from DB
            db_data = [np.array(i[0]).squeeze() for i in stmall]
            entity = [i[1] for i in stmall]
            db_id = [i[2] for i in stmall]
            # Convert to a dataframe
            to_return = pd.DataFrame(data={'db_id': db_id, 'data': db_data, 'entity': entity}, index=entity)
            if self._use_cache:
                # Update cache
                self._db_cache.update(to_return['db_id'].to_numpy(), to_return)
                # Merge cached and hit
                if len(cached) > 0:
                    cached.index = cached['entity']
                    to_return = to_return.append(cached)
            # Return
            return to_return

    def collect(self):
        """Get all measurements"""
        return self._get(range(len(self.db_ids)))

    def mask(self, id_set):
        """Create a set from the intersection with other ids"""
        if type(id_set) is MeasurementCollection:
            id_set = id_set.entity_ids
        elif type(id_set) is list:
            pass
        elif type(id_set) is pd.DataFrame:
            id_set = id_set.index.tolist()
        else:
            raise TypeError(
                'Mask must be passed either a list of entity IDs, another MeasurementCollection or a Measurement '
                'dataframe')
            # Create an intersection on entity IDs
        (intersection, id1, id2) = np.intersect1d(self.entity_ids, id_set, return_indices=True)
        # Filter measurement IDs
        measurement_ids = [self.db_ids[i] for i in id1]
        # Return a new filtered measurement collection
        return MeasurementCollection(measurement_ids=measurement_ids, entity_ids=intersection)

    def __next__(self):
        """To iterate"""
        self.cursor = self.cursor + 1
        if self.cursor == len(self.db_ids):
            self.cursor = 0
            raise StopIteration()
        return self._get([self.cursor])

    def __iter__(self):
        return self


#################################################################
#   PostProcess
#################################################################

class PostProcess:
    """A wrapper around a MeasurementCollection or another PostProcess function to cleanly add line-by-line
    processing. """
    mc = None
    func = None
    _cursor = 0
    data_column = 'data'

    def __init__(self, mc=None):
        """Initialise the PostProcess class by passing a measurementCollection or a PostProcess class"""
        if type(mc) in [MeasurementCollection, PostProcess]:
            self.mc = mc
            if type(mc) is MeasurementCollection:
                self.data_column = 'data'
            elif type(mc) is PostProcess:
                self.data_column = 'processed'
        elif type(mc) is None:
            return
        else:
            raise TypeError(
                'PostProcess must be initialised with a MeasurementCollection or a PostProcess class, not a  "{}"'
                    .format(type(mc)))

    def __repr__(self):
        """Representation"""
        if self.func:
            function_name = self.func.__name__
        else:
            function_name = 'None'
        if self.mc:
            dataset_name = repr(self.mc)
        else:
            dataset_name = 'None'
        return "Postprocessing function [{}] attached to {}".format(function_name, dataset_name)

    def __len__(self):
        """Return number of underlying datasets"""
        return len(self.mc)

    def set_function(self, func):
        """Set the function. This should take the underlying data type as an input and return something based on
        this. """
        self.func = func

    def set_data(self, mc):
        """Set the datasource, either a MeasurementClass or a PostProcess class."""
        self.mc = mc
        if type(mc) is MeasurementCollection:
            self.data_column = 'data'
        elif type(mc) is PostProcess:
            self.data_column = 'processed'

    def __next__(self):
        """"To iterate"""
        self._cursor = self._cursor + 1
        if self._cursor == len(self.mc.db_ids):
            self._cursor = 0
            raise StopIteration()
        processed = self.mc._get([self._cursor])
        # This is a dataframe
        processed['processed'] = self.func(processed[self.data_column])
        return processed

    def __iter__(self):
        return self

    def collect(self):
        """Get all measurements"""
        to_ret = self.mc.collect()
        to_ret['processed'] = to_ret.apply(lambda row: self.func(row[self.data_column]), axis=1)
        return to_ret

    def sample(self, k=1):
        """Return a subset of k processed sets"""
        to_ret = self.mc.sample(k=k)
        to_ret['processed'] = to_ret.apply(lambda row: self.func(row[self.data_column]), axis=1)
        return to_ret
