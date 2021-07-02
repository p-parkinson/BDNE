# Definition of data structures for BDNE project
# Author : Patrick Parkinson <patrick.parkinson@manchester.ac.uk>

# Imports for type-annotations
from __future__ import annotations
from typing import Tuple, List, Dict, Union, Callable, Any

# Import random to sample from the sets
import random

# Import numpy as part of the type-annotations
import numpy as np

# Import pandas to output the data as a dataframe
import pandas as pd

# Import Mapping to implement the experimental metadata as a mapping class
from collections.abc import Mapping

# Import the core BDNE ORM and configuration to deal with the database
import BDNE.db_orm as db
import BDNE.config as cfg
from BDNE.config import db_batch_size


#################################################################
#   A cache class for storing data locally
#################################################################

class DBCache:
    """A basic cache class for database IDs- never kicks out old data unless told to"""
    # Store the data in pd.DataFrame
    _cache: pd.DataFrame

    def __init__(self) -> None:
        """Set up pandas dataframe to store data internally"""
        self._cache = pd.DataFrame()

    def clear(self) -> None:
        """Empty the cache"""
        self._cache = pd.DataFrame()

    def __call__(self, ids: List[int]) -> Tuple[List[int], pd.DataFrame]:
        """Convenience function to retrieve a list of results"""
        return self.check(ids)

    def check(self, ids: List[int]) -> Tuple[List[int], pd.DataFrame]:
        """Look for hits with supplied ids, must be unique index"""
        if len(ids) == 0:
            return [], pd.DataFrame()
        ids = np.array(ids)
        # Get from cache
        cached = self._cache[self._cache.index.isin(ids)]
        # List not found items to be read in
        not_found = np.setdiff1d(ids, cached.index.to_numpy()).tolist()
        return not_found, cached

    def update(self, ids: np.Array, data: pd.DataFrame) -> None:
        """Update the cache for the ids provided with the data provided"""
        # Convert to integer
        ids = ids.astype('int')
        # Make sure not to update existing data
        missing = np.setdiff1d(ids, self._cache.index.to_numpy())
        # update index
        old_index = data.index
        data.index = ids
        # Create new dataframe
        self._cache = self._cache.append(data[data.index.isin(missing)])
        # Restore
        data.index = old_index

    def __len__(self) -> pd.Series:
        """Return the amount of memory used by the cache."""
        return self._cache.memory_usage(deep=True)


#################################################################
#   A single entity class
#################################################################
class Entity:
    """Class to store all of the data for a given entity.
        Typical Usage:
        w = Entity(1000);
        print(w);"""
    # The database ID of this entity
    db_id: int = None
    # The sample ID (with additional information about the wider sample the entity is from)
    _sample_id: int = None
    # An internal container for the experimental data associated with this object
    experiment_container = []

    def __repr__(self) -> str:
        """Return information about this entity, including all experiments (if cached)."""
        r = "{} ID={}".format(self.__class__.__name__, self.db_id)
        if len(self.experiment_container) > 0:
            r += " {}".format([i[0] for i in self.experiment_container])
        return r

    def __init__(self, db_id: int = None) -> None:
        """Initialise the entity class as empty, or with an entity id."""
        if db_id is None:
            return
        # ID given
        self.db_id = db_id

    def sample(self) -> Dict[str, str]:
        """Return data about the sample that this entity is associated with."""
        if self._sample_id is None:
            self._sample_id = cfg.session.query(db.Entity.sampleID).filter(db.Entity.ID == self.db_id).first()[0]
        # Set up database query to retrieve
        stm = cfg.session.query(db.Sample.ID, db.Sample.supplier, db.Sample.material, db.Sample.preparation_date,
                                db.Sample.preparation_method, db.Sample.substrate).filter(
            db.Sample.ID == self._sample_id).first()
        # Zip to dictionary
        keys = ['ID', 'Supplier', 'Material', 'Preparation_date', 'Preparation_method', 'Substrate']
        return dict(zip(keys, stm))

    def populate_from_db(self) -> None:
        """Retrieve all experiments associated with this entity ID"""
        stm = cfg.session.query(db.Experiment.type, db.Measurement.ID).join(db.Measurement).join(db.Object).\
            join(db.Entity).filter(db.Entity.ID == self.db_id)
        # Check whether this entity exists
        if not stm.all():
            raise KeyError('No Entity exists with ID {}'.format(self.db_id))
        self.experiment_container = stm.all()

    def experiments(self) -> List[str]:
        """List all experiments associated with this entity"""
        if not self.experiment_container:
            self.populate_from_db()
        return [i[0] for i in self.experiment_container]

    # TODO: Find type hint for sqlalchemy session.query
    def get(self, experiment: Union[int, str]):
        """Get a single experiment associated with this entity by experiment number or name"""
        # Check if we have downloaded experiment list yet
        if not self.experiment_container:
            self.populate_from_db()
        # Check type of experiment
        if type(experiment) is int:
            exp_id = self.experiment_container[experiment][1]
        elif type(experiment) is str:
            exp_id = [i[1] for i in self.experiment_container if i[0] == experiment]
            # Check how many datasets are associated with this
            if len(exp_id) == 0:
                raise KeyError('Experiment {} not present for Entity ID {}'.format(experiment, self.db_id))
            elif len(exp_id) == 1:
                exp_id = exp_id[0]
            else:
                raise KeyError('Experiment {} ambiguous for Entity ID {}'.format(experiment, self.db_id))
        else:
            raise TypeError('Experiment must be defined as an integer or a string')
        # Retrieve experiment results from database
        stm = cfg.session.query(db.Measurement.data).filter(db.Measurement.ID == exp_id)
        if len(stm.all()) == 1:
            return stm.all()[0][0]
        else:
            raise KeyError('Measurement ID {} not found in database'.format(exp_id))


#################################################################
#   EntityCollection (a collection of Entities)
#################################################################

class EntityCollection:
    """A collection of entities.
    Lazy handling, stores only db_ids for the entities and returns either an entity, a set of entities,
    or a set of measurements.
      Typical usage:
      ``w = EntityCollection();
      w.load_sample(25);``"""

    # Database IDs associated with entities in this set
    db_ids: List[int] = []
    # Cursor to use as iterator
    cursor: int = -1

    def __repr__(self) -> str:
        """Return string describing collection"""
        if self.db_ids:
            return "{} IDs={}".format(self.__class__.__name__, len(self.db_ids))
        else:
            return f'Empty {self.__class__.__name__}'

    def __init__(self, start_id: List[int] = None) -> None:
        """Set up wire collection, either blank or with a set of initial entity IDs."""
        self.db_ids = start_id

    def __len__(self) -> int:
        """The number of entities in this collection"""
        return len(self.db_ids)

    def __del__(self) -> None:
        # Clear up the data in collections
        pass

    def __getstate__(self) -> List[int]:
        """Select what gets pickled"""
        return self.db_ids

    def __setstate__(self, state: List[int]) -> None:
        """Only restore db_ids"""
        self.db_ids = state

    def load_sample(self, sample_id: int) -> None:
        """Load a sample ID into the EntityCollection class"""
        stm = cfg.session.query(db.Entity.ID).filter(db.Entity.sampleID == sample_id)
        self.db_ids = [i[0] for i in stm.all()]
        if not self.db_ids:
            raise Warning('No Entities found with sample ID {}'.format(sample_id))

    def load_entity_group(self, entity_group_id: int) -> None:
        """Load an entityGroup ID into the EntityCollection class"""
        stm = cfg.session.query(db.EntityGroupEntity.entityID).filter(db.EntityGroup.ID == entity_group_id)
        self.db_ids = [i[0] for i in stm.all()]
        # Check if any entities are returned
        if not self.db_ids:
            raise Warning('No Entities found with sample ID {}'.format(entity_group_id))

    def sample(self, number_to_sample: int = 0) -> Union[Entity, EntityCollection]:
        """Return a random subset of k entities from the EntityCollection."""
        if number_to_sample > 0:
            wid = random.sample(self.db_ids, k=number_to_sample)
            # Select - return either an Entity or a EntityCollection
            if len(wid) == 1:
                return Entity(wid[0])
            else:
                return EntityCollection(wid)
        else:
            raise TypeError('Argument to sample must be an integer.')

    def mask(self, id_set: Union[EntityCollection, MeasurementCollection]) -> EntityCollection:
        """Create a new entity set from an intersection with other entity ids"""
        if type(id_set) is EntityCollection:
            id_set = id_set.db_ids
        if type(id_set) is MeasurementCollection:
            id_set = id_set.entity_ids
        else:
            raise TypeError('Mask must be passed as either a MeasurementCollection or another EntityCollection')
        # Create an intersection between the local IDs and the remote ID set
        intersection = set(self.db_ids).intersection(id_set)
        return EntityCollection(list(intersection))

    def logical_mask(self, mask: np.Array) -> EntityCollection:
        """Create a new wire collection using a logical mask"""
        new_ids = np.array(self.db_ids)[mask].tolist()
        return EntityCollection(new_ids)

    def get_entity(self, id: int) -> Entity:
        """Return a single entity"""
        return Entity(self.db_ids[id])

    def get_measurement(self, experiment_name: str) -> MeasurementCollection:
        """Return a MeasurementCollection (when a string is passed)"""
        # Make a copy of the db_ids
        all_db_ids = self.db_ids.copy()
        # Create empty lists to hold the ids
        measurement_ids = []
        entity_ids = []
        # Pop ids to get
        while len(all_db_ids) > 0:
            # For final batch, take all. For earlier batches, take
            if len(all_db_ids) < db_batch_size:
                sub_query = all_db_ids
                all_db_ids = []
            else:
                sub_query = all_db_ids[0:db_batch_size]
                all_db_ids = all_db_ids[db_batch_size:]
            # Create statement
            stm = cfg.session.query(db.Measurement.ID, db.Entity.ID).select_from(db.Measurement). \
                join(db.Object).join(db.Entity).join(db.Experiment). \
                filter(db.Entity.ID.in_(sub_query), db.Experiment.type == experiment_name)
            # Execute statement
            ret = stm.all()
            # Add returned to lists
            measurement_ids.extend([i[0] for i in ret])
            entity_ids.extend([i[1] for i in ret])
        # Return a MeasurementCollection
        return MeasurementCollection(measurement_ids=measurement_ids, entity_ids=entity_ids)

    def __next__(self) -> Entity:
        """To iterate over each entity in the Collection"""
        self.cursor = self.cursor + 1
        # Check for end of list
        if self.cursor == len(self.db_ids):
            self.cursor = 0
            raise StopIteration()
        return self.get_entity(self.cursor)

    def __iter__(self) -> EntityCollection:
        # Return self
        return self

    def __add__(self, other: EntityCollection) -> EntityCollection:
        """Combine two entityCollections and return a merged set"""
        return EntityCollection(self.db_ids + other.db_ids)


#################################################################
#   MeasurementCollection
#################################################################

class MeasurementCollection:
    """A class to hold a collection of related measurement.
    Uses lazy loading, holding only the database IDs and associated entity IDs until a get()
    or collect() is issued.
        Typical Usage:
        w = EntityCollection();
        w.load_entity_group(4);
        e = w.get_measurement('spectra'); # A MeasurementCollection"""
    # Database IDs for the measurements
    db_ids: List[int] = []
    # Associated entity IDs
    entity_ids: List[int] = []
    # Cursor for use as an iterator
    cursor: int = -1
    # Internal link to cache
    _db_cache: DBCache = DBCache()
    # Cache switch
    _use_cache: bool = True

    def __init__(self, measurement_ids: Union[np.array, List[int]] = None,
                 entity_ids: Union[np.array, List[int]] = None) -> None:
        """Initialise with a list of measurement_IDs and entity_ids"""
        if len(measurement_ids) == len(entity_ids):
            self.db_ids = measurement_ids
            self.entity_ids = entity_ids
        else:
            raise RuntimeError('Both measurement_id and entity_id must be provided with the same length.')

    def __repr__(self) -> str:
        """Return string representation"""
        if len(self.db_ids) > 0:
            return "{} IDs={}".format(self.__class__.__name__, len(self.db_ids))
        else:
            return f"Empty {self.__class__.__name__}"

    def __len__(self) -> int:
        """Return the number of measurements in this collection"""
        return len(self.db_ids)

    def __del__(self) -> None:
        # Remove the instance from memory and remove the associated Collection data
        pass

    def __getstate__(self) -> dict:
        """Only store/pickle entity and db_ids"""
        return {'db_ids': self.db_ids, 'entity_ids': self.entity_ids}

    def __setstate__(self, state: dict) -> None:
        """Only restore db_ids and entity ids"""
        self.db_ids = state['db_ids']
        self.entity_ids = state['entity_ids']

    def sample(self, number: int = 1) -> pd.DataFrame:
        """Get a random selection of 'number' measurements"""
        selected = random.choices(range(len(self.db_ids)), k=number)
        return self._get(selected)

    def _get(self, n: Union[range, list]) -> pd.DataFrame:
        """A cached function to return measurements from the set."""
        # Convert ranges to a list
        if type(n) is range:
            n = list(n)
        # Check if a list passed (must be)
        if type(n) is not list:
            raise NotImplementedError('n must be a list')
        # Convert list to numpy array
        n = np.array(n)
        # Range check
        if np.any(n > len(self.db_ids)) or np.any(n < 0):
            raise KeyError('Index must be in range 0 to {}'.format(len(self.db_ids)))
        # Convert indices to db_ids
        to_get = [self.db_ids[i] for i in n]
        # If zero length, return empty
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
            # Collect any remaining datasets rest
            if len(to_get) > 0:
                # TODO: Remove temporary table and use batched retrieve for large sets
                # Initialise
                db_data, entity, db_id, exp_id = [], [], [], []
                while len(to_get) > 0:
                    if len(to_get) < db_batch_size:
                        sub_query = to_get
                        to_get = []
                    else:
                        sub_query = to_get[0:db_batch_size]
                        to_get = to_get[db_batch_size:]
                    # Assemble the query
                    stm = cfg.session.query(db.Measurement.data, db.Object.entity_id,
                                            db.Measurement.ID, db.Measurement.experiment_ID).\
                        join(db.Object).filter(db.Measurement.ID.in_(sub_query))
                    # Collection from database
                    query_result = stm.all()
                    # Format from DB
                    db_data.extend([np.array(i[0]).squeeze() for i in query_result])
                    entity.extend([i[1] for i in query_result])
                    db_id.extend([i[2] for i in query_result])
                    exp_id.extend([i[3] for i in query_result])
                # Convert to a dataframe
                to_return = pd.DataFrame(
                    data={'db_id': db_id, 'entity': entity, 'experiment_id': exp_id, 'data': db_data},
                    index=entity)
            else:
                # Empty dataframe from database - all cached
                to_return = pd.DataFrame(columns=['db_id', 'entity', 'experiment_id', 'data'])
            # Assemble results from cache and direct database access
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
        """Get all measurements and return as a dataframe"""
        return self._get(range(len(self.db_ids)))

    def collect_as_matrix(self) -> np.array:
        """Get all measurements as an n x m array, where n entities with m data points per measurement"""
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
        # Check if final
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
    """A wrapper around a MeasurementCollection or another PostProcess function to cleanly add
    line-by-line processing.
        Typical Usage:
        w = EntityCollection()
        w.load_entity_group(4)
        spectra = w.get_measurement('spectra')
        PL = PostProcess(spectra)
        PL.set_function(np.sum)"""

    # Define the function type
    function_type = Callable[[pd.Series], Any]
    # Define the underlying dataset
    mc: Union[PostProcess, MeasurementCollection] = None
    # Handle to function
    func: function_type = None
    # Cursor for iterating
    _cursor: int = 0
    # Column name for the data in the pandas set
    data_column: str = 'data'

    def __init__(self, mc: Union[MeasurementCollection, PostProcess] = None) -> None:
        """Initialise the PostProcess class by passing a measurementCollection or a PostProcess class"""
        if type(mc) in [MeasurementCollection, PostProcess]:
            self.mc = mc
            # If we wrap a MeasurementCollection then base on the "data" column, else on the "processed" column
            if type(mc) is MeasurementCollection:
                self.data_column = 'data'
            elif type(mc) is PostProcess:
                self.data_column = 'processed'
        elif mc is None:
            return
        else:
            raise TypeError(
                'PostProcess must be initialised with a MeasurementCollection or a PostProcess class, not a  "{}"'
                .format(type(mc))
            )

    def __repr__(self) -> str:
        """Represent as string"""
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
        # Get the underlying data
        to_ret = self.mc.collect()
        # Return through an "apply" function
        to_ret['processed'] = to_ret.apply(lambda row: self.func(row[self.data_column]), axis=1)
        return to_ret

    def sample(self, number=1) -> pd.DataFrame:
        """Return a subset of k processed sets"""
        to_ret = self.mc.sample(number=number)
        to_ret['processed'] = to_ret.apply(lambda row: self.func(row[self.data_column]), axis=1)
        return to_ret

    def collect_as_matrix(self) -> np.array:
        """Return the full output as a numpy array"""
        return np.stack(self.collect()['processed'])


#################################################################
#   Experimental metadata
#   TODO: Allow mutable metadata - addition to database
#################################################################

class ExperimentMetadata(Mapping):
    """Container for experimental metadata. Not currently possible to change metadata"""
    # Define type for the value
    value_type = Union[np.array, List[int], int]
    # Define type for the ID
    id_type = Union[int, np.int, np.int64, pd.DataFrame]
    # Experimental ID to associated metadata with
    experiment_id: id_type = None
    # Internal dictionary to store metadata
    _internal_mapping = {}

    def __getitem__(self, k: str) -> value_type:
        """Getter from internal mapping"""
        return self._internal_mapping[k]

    def __len__(self) -> int:
        """Returns number of metadata entries"""
        return len(self._internal_mapping)

    def __iter__(self):
        # Iteration not possible
        pass

    def __repr__(self) -> str:
        """String representation"""
        return repr(self._internal_mapping)

    def keys(self):
        """Return keys of mapping"""
        return self._internal_mapping.keys()

    def load_values(self, experiment_id: int = None) -> None:
        """Refresh from database"""
        if experiment_id:
            self.experiment_id = experiment_id
        stm = cfg.session.query(db.Metadata.key, db.Metadata.value) \
            .filter(db.Metadata.experiment_id == self.experiment_id).all()
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
            eid = cfg.session.query(db.Measurement.experiment_ID).filter(db.Measurement.ID == measurement_id).all()
            if len(eid) == 1:
                self.experiment_id = eid[0]
            else:
                raise ValueError('Could not find unique experiment associated with measurement ID')
        else:
            raise KeyError('Must pass experiment ID or measurement ID to associate with the meta-information')
        # Send to load
        self.load_values()
