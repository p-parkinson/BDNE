# Definition of data structures for BDNE project
# Author : Patrick Parkinson <patrick.parkinson@manchester.ac.uk>

# For type annotation
from __future__ import annotations
from typing import Tuple, List, Dict, Union, Callable, Any

# For sampling from set
import random

# For conversion
import numpy as np

# For datasets
import pandas as pd

# For Experimental metadata as a mapping
from collections.abc import Mapping

# Import for database
from BDNE.db_orm import *
from BDNE import session


#################################################################
#   A cache class for storing data locally
#################################################################

class DBCache:
    """A simple cache class - never kicks out old data unless told to"""
    _cache: pd.DataFrame

    def __init__(self) -> None:
        """Set up pandas dataframe to store data"""
        self._cache = pd.DataFrame()

    def clear(self) -> None:
        """Empty the cache"""
        self._cache = pd.DataFrame()

    def __call__(self, ids: List[int]) -> Tuple[List[int], pd.DataFrame]:
        """Convenience function"""
        return self.check(ids)

    def check(self, ids: List[int]) -> Tuple[List[int], pd.DataFrame]:
        """Look for hits with ids, must be unique"""
        if len(ids) == 0:
            return [], pd.DataFrame()
        ids = np.array(ids)
        # Get from cache
        cached = self._cache[self._cache.index.isin(ids)]
        # List unfound items to be read in
        unfound: List[int] = np.setdiff1d(ids, cached.index.to_numpy()).tolist()
        return unfound, cached

    def update(self, ids: np.Array, data: pd.DataFrame) -> None:
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

    def __len__(self) -> pd.Series:
        return self._cache.memory_usage(deep=True)


#################################################################
#   A single wire class
#################################################################
class Wire:
    """All data for a unique single element."""
    db_id: int = None
    _sample_id: int = None
    experiment_container = []

    def __repr__(self) -> str:
        """Representation"""
        r = "{} ID={}".format(self.__class__.__name__, self.db_id)
        if len(self.experiment_container) > 0:
            r += " {}".format([i[0] for i in self.experiment_container])
        return r

    def __init__(self, db_id: int = None) -> None:
        """Initialise the wire class as empty or with a db_id."""
        if db_id is None:
            return
        # ID given
        self.db_id = db_id

    def sample(self) -> Dict[str, str]:
        """Return sample ID"""
        if self._sample_id is None:
            self._sample_id = session.query(Entity.sampleID).filter(Entity.ID == self.db_id).first()[0]
        stm = session.query(Sample.ID, Sample.supplier, Sample.material, Sample.preparation_date,
                            Sample.preparation_method, Sample.substrate).filter(Sample.ID == self._sample_id).first()
        keys = ['ID', 'Supplier', 'Material', 'Preparation_date', 'Preparation_method', 'Substrate']
        return dict(zip(keys, stm))

    def populate_from_db(self) -> None:
        """Retrieve all experiments associated with this nanowire ID"""
        stm = session.query(Experiment.type, Measurement.ID).join(Measurement).join(Object).join(Entity).filter(
            Entity.ID == self.db_id)
        if not stm.all():
            raise KeyError('No Entity exists with ID {}'.format(self.db_id))
        self.experiment_container = stm.all()

    def experiments(self) -> List[str]:
        """List all experiments in this wire"""
        if not self.experiment_container:
            self.populate_from_db()
        return [i[0] for i in self.experiment_container]

    # TODO: Find type hint for sqlalchemy session.query
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

    db_ids: List[int] = []
    cursor: int = -1

    def __repr__(self) -> str:
        """Representation"""
        return "{} IDs={}".format(self.__class__.__name__, len(self.db_ids))

    def __init__(self, start_id: List[int] = None) -> None:
        """Set up wire collection, either blank or with a set of initial entity IDs."""
        self.db_ids = start_id

    def __len__(self) -> int:
        """The number of wires in this collection"""
        return len(self.db_ids)

    def load_sample(self, sample_id: int) -> None:
        """Load a sample ID into the WireCollection class"""
        stm = session.query(Entity.ID).filter(Entity.sampleID == sample_id)
        self.db_ids = [i[0] for i in stm.all()]
        if not self.db_ids:
            raise Warning('No wires found with sample ID {}'.format(sample_id))

    def load_entity_group(self, entity_group_id: int) -> None:
        """Load an entityGroup ID into the WireCollection class"""
        # TODO: implement entity_group loading
        stm = session.query(EntityGroupEntity.entityID).filter(EntityGroup.ID == entity_group_id)
        self.db_ids = [i[0] for i in stm.all()]
        if not self.db_ids:
            raise Warning('No wires found with sample ID {}'.format(entity_group_id))

    def sample(self, number_to_sample: int = 0) -> Union[Wire, WireCollection]:
        """Return a random subset of k entities from the WireCollection."""
        # TODO: Deal with different return types
        if number_to_sample > 0:
            wid = random.choices(self.db_ids, k=number_to_sample)
            if len(wid) == 1:
                return Wire(wid[0])
            else:
                return WireCollection(wid)
        else:
            raise TypeError('Argument to sample must be an integer.')

    def mask(self, id_set: Union[WireCollection, MeasurementCollection]) -> WireCollection:
        """Create a new entity set from an intersection with other entity ids"""
        if type(id_set) is WireCollection:
            id_set = id_set.db_ids
        if type(id_set) is MeasurementCollection:
            id_set = id_set.entity_ids
        else:
            raise TypeError('Mask must be passed either a MeasurementCollection or another WireCollection')
        # Create an intersection (where
        intersection = set(self.db_ids).intersection(id_set)
        return WireCollection(list(intersection))

    def logical_mask(self, mask: np.Array) -> WireCollection:
        """Create a new wire collection using a logical mask"""
        new_ids = np.array(self.db_ids)[mask].tolist()
        return WireCollection(new_ids)

    def get_wire(self, wire_id: int) -> Wire:
        """Return either a single entity (when enumerated)"""
        return Wire(self.db_ids[wire_id])

    def get_measurement(self, experiment_name: str) -> MeasurementCollection:
        """Return a MeasurementCollection (when a string is passed)"""
        if type(experiment_name) is str:
            # Return a measurement collection with associated entity (to backreference)
            stm = session.query(Measurement.ID, Entity.ID).select_from(Measurement). \
                join(Object).join(Entity).join(Experiment). \
                filter(Entity.ID.in_(self.db_ids), Experiment.type == experiment_name)
            ret = stm.all()
            return MeasurementCollection([i[0] for i in ret], [i[1] for i in ret])

    def __next__(self) -> Wire:
        """To iterate over each wire in the Collection"""
        self.cursor = self.cursor + 1
        if self.cursor == len(self.db_ids):
            self.cursor = 0
            raise StopIteration()
        return self.get_wire(self.cursor)

    def __iter__(self) -> WireCollection:
        return self

    def __add__(self, other: WireCollection) -> WireCollection:
        """Combine two entityCollections and return a merged set"""
        return WireCollection(self.db_ids + other.db_ids)


#################################################################
#   MeasurementCollection
#################################################################

class MeasurementCollection:
    """A class to hold a collection of related measurement.
    Uses lazy loading, holding only the database IDs and associated entity IDs until a get() or collect() is issued."""
    db_ids: List[int] = []
    entity_ids: List[int] = []
    cursor: int = -1
    _db_cache: DBCache = DBCache()
    _use_cache: bool = True

    def __init__(self, measurement_ids: Union[np.array, List[int]] = None, entity_ids: Union[np.array, List[int]] = None) -> None:
        """Initialise with a list of measurement_IDs and entity_ids"""
        if type(measurement_ids) is list:
            if len(measurement_ids) == len(entity_ids):
                self.db_ids = measurement_ids
                self.entity_ids = entity_ids
            else:
                raise RuntimeError('Both measurement_id and entity_id must be provided with the same length.')

    def __repr__(self) -> str:
        """Representation"""
        return "{} IDs={}".format(self.__class__.__name__, len(self.db_ids))

    def __len__(self) -> int:
        """Return the number of measurements in this collection"""
        return len(self.db_ids)

    def sample(self, number: int = 1) -> pd.DataFrame:
        """Get a random selection of k measurements"""
        selected = random.choices(range(len(self.db_ids)), k=number)
        return self._get(selected)

    def _get(self, n: Union[range, list]) -> pd.DataFrame:
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
            return pd.DataFrame()
        else:
            # Multiple datasets to return
            # Need to check cache
            if self._use_cache:
                (to_get, cached) = self._db_cache.check(to_get)
            else:
                cached = None
            # Collect the rest
            if len(to_get) > 0:
                stm = session.query(Measurement.data, Object.entity_id, Measurement.ID, Measurement.experiment_ID).join(
                    Object).filter(
                    Measurement.ID.in_(to_get))
                stmall = stm.all()
                # Format from DB
                db_data = [np.array(i[0]).squeeze() for i in stmall]
                entity = [i[1] for i in stmall]
                db_id = [i[2] for i in stmall]
                exp_id = [i[3] for i in stmall]
                # Convert to a dataframe
                to_return = pd.DataFrame(
                    data={'db_id': db_id, 'entity': entity, 'experiment_id': exp_id, 'data': db_data},
                    index=entity)
            else:
                to_return = pd.DataFrame(columns=['db_id', 'entity', 'experiment_id', 'data'])
            if self._use_cache:
                # Update cache
                self._db_cache.update(to_return['db_id'].to_numpy(), to_return)
                # Merge cached and hit
                if len(cached) > 0:
                    cached.index = cached['entity']
                    to_return = to_return.append(cached)
            # Return
            return to_return

    def collect(self) -> pd.DataFrame:
        """Get all measurements"""
        return self._get(range(len(self.db_ids)))

    def collect_as_matrix(self) -> np.array:
        """Get all measurements as an n x m array, where n wires with m datapoints per measurement"""
        return np.stack(self.collect()['data'])

    def mask(self, id_set: Union[pd.DataFrame, MeasurementCollection, list]) -> MeasurementCollection:
        """Create a set from the intersection with other ids"""
        if type(id_set) is MeasurementCollection:
            id_set = id_set.entity_ids
        elif type(id_set) is list:
            pass
        elif type(id_set) is pd.DataFrame:
            id_set = id_set.index.to_list()
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

    def __next__(self) -> pd.DataFrame:
        """To iterate"""
        self.cursor = self.cursor + 1
        if self.cursor == len(self.db_ids):
            self.cursor = 0
            raise StopIteration()
        return self._get([self.cursor])

    def __iter__(self) -> MeasurementCollection:
        return self


#################################################################
#   PostProcess
#################################################################

class PostProcess:
    """A wrapper around a MeasurementCollection or another PostProcess function to cleanly add line-by-line
    processing. """
    function_type = Callable[[pd.Series], Any]
    mc: Union[PostProcess, MeasurementCollection] = None
    func: function_type = None
    _cursor: int = 0
    data_column: str = 'data'

    def __init__(self, mc: Union[MeasurementCollection, PostProcess] = None) -> None:
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
                .format(type(mc))
            )

    def __repr__(self) -> str:
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

    def __len__(self) -> int:
        """Return number of underlying datasets"""
        return len(self.mc)

    def set_function(self, func: function_type) -> None:
        """Set the function. This should take the underlying data type as an input and return something based on
        this. """
        self.func = func

    def set_data(self, mc: Union[MeasurementCollection, PostProcess]) -> None:
        """Set the datasource, either a MeasurementClass or a PostProcess class."""
        self.mc = mc
        if type(mc) is MeasurementCollection:
            self.data_column = 'data'
        elif type(mc) is PostProcess:
            self.data_column = 'processed'

    def __next__(self) -> Any:
        """"To iterate"""
        self._cursor = self._cursor + 1
        if self._cursor == len(self.mc.db_ids):
            self._cursor = 0
            raise StopIteration()
        processed = self.mc._get([self._cursor])
        # This is a dataframe
        processed['processed'] = self.func(processed[self.data_column])
        return processed

    def __iter__(self) -> PostProcess:
        return self

    def collect(self) -> pd.DataFrame:
        """Get all measurements"""
        to_ret = self.mc.collect()
        to_ret['processed'] = to_ret.apply(lambda row: self.func(row[self.data_column]), axis=1)
        return to_ret

    def sample(self, number=1) -> pd.DataFrame:
        """Return a subset of k processed sets"""
        to_ret = self.mc.sample(number=number)
        to_ret['processed'] = to_ret.apply(lambda row: self.func(row[self.data_column]), axis=1)
        return to_ret

    def collect_as_matrix(self) -> np.array:
        return np.stack(self.collect()['processed'])


#################################################################
#   Experimental metadata
#   TODO: Allow mutable metadata - addition to database
#################################################################

class ExperimentMetadata(Mapping):
    """Container for experimental metadata. Not currently possible to change metadata"""
    value_type = Union[np.array, List[int], int]
    id_type = Union[int, np.int, np.int64, pd.DataFrame]
    experiment_id: id_type = None
    _internal_mapping = {}

    def __getitem__(self, k: str) -> value_type:
        return self._internal_mapping[k]

    def __len__(self) -> int:
        return len(self._internal_mapping)

    def __iter__(self):
        pass

    def __repr__(self) -> str:
        return repr(self._internal_mapping)

    def keys(self):
        return self._internal_mapping.keys()

    def load_values(self, experiment_id: int = None) -> None:
        """Refresh from database"""
        if experiment_id:
            self.experiment_id = experiment_id
        stm = session.query(Metadata.key, Metadata.value) \
            .filter(Metadata.experiment_id == self.experiment_id).all()
        # Dictionary comprehension to internal
        if len(stm) > 0:
            self._internal_mapping = {k: v for (k, v) in stm}
        else:
            self._internal_mapping = {}

    def __init__(self, experiment_id: id_type = None, measurement_id: int = None) -> None:
        """Initialise the class to either an experimental ID or a measurement ID associated with an experiment."""
        if experiment_id:
            if type(experiment_id) in [int, np.int, np.int64]:
                self.experiment_id = experiment_id
            elif type(experiment_id) is pd.DataFrame:
                self.experiment_id = experiment_id.experiment_id.iloc[0]
            else:
                raise KeyError("Experiment ID should be an integer or a dataframe")
        elif measurement_id:
            eid = session.query(Measurement.experiment_ID).filter(Measurement.ID == measurement_id).all()
            if len(eid) == 1:
                self.experiment_id = eid[0]
            else:
                raise ValueError('Could not find unique experiment associated with measurement ID')
        else:
            raise KeyError('Must pass experiment ID or measurement ID to associate with the meta-information')
        # Send to load
        self.load_values()
