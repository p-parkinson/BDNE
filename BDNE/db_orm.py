# SQLAlchemy-based ORM model for the BDNE project
# Author : Patrick Parkinson <patrick.parkinson@manchester.ac.uk>

# Zlib used for compression for data
import zlib

# Datetime for column definitions
from datetime import datetime

# Numpy for handling numeric data
import numpy as np

# SQLAlchemy used for database connection
import sqlalchemy.types as types
from sqlalchemy import Integer, ForeignKey, Text, String, Column, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

# Matlab serialise for numpy->matlab and matlab->numpy conversion
from BDNE.matlab_serialise import serialise, deserialise
# Global session container
import BDNE.config

# Create base autoloader
decBase = declarative_base()


############################################################################
# Custom class for data in database
############################################################################
def connect(big_query=None, mysql=None) -> Session:
    """Connect to the underlying data source, either via big_query or mysql"""
    # Test with bigquery
    if big_query is not None:
        engine = create_engine(big_query['bigquery_uri'],
                               credentials_path=big_query['credentials_path'])
    # Test with mysql
    elif mysql is not None:
        engine = create_engine('mysql+mysqlconnector://%s:%s@%s:%s/bdne' %
                               (mysql['user'], mysql['password'], mysql['server'], mysql['port']))
    else:
        raise(KeyError('Must specify either big_query or mysql configuration.'))
    # Create session
    BDNE.config.session = Session(engine)
    # Return session
    return BDNE.config.session


############################################################################
# Custom class for data in database
############################################################################

# Set up serialise/deserialise
class MATLAB(types.TypeDecorator):
    """Converts to and from hlp_serialised MATLAB form on the fly"""

    # Define underlying data type
    impl = types.String

    def process_bind_param(self, value, dialect):
        """Run serialise (forward)"""
        return serialise(value)

    def process_result_value(self, value, dialect) -> np.ndarray:
        """Run deserialise (backwards)"""
        # Fix for odd error in LargeBinary (see
        # https://github.com/sqlalchemy/sqlalchemy/issues/5073#issuecomment-582953477)
        # where incorrect encoding appears

        if isinstance(value, str):
            # Encode as utf-8
            value = bytes(value, 'utf-8')
        elif value is not None:
            # Encode as bytes
            value = bytes(value)

        # Check for empty data
        if len(value) == 0:
            return np.empty(0)

        # Check for compression
        if value[0] == 201:
            # Dataset is compressed. Cut first value, decompress with zlib
            value = zlib.decompress(value[1:])

        # Deserialise via MATLAB
        value = deserialise(value)
        return value


############################################################################
# Define all tables in the database
############################################################################

class Object(decBase):
    """Object SQLAlchemy Table"""
    __tablename__ = 'object'

    ID = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(Integer, ForeignKey('entity.ID'))
    created = Column(DateTime, default=datetime.now())

    entity = relationship("Entity", back_populates="objects")
    measurement = relationship("Measurement", back_populates="object")


class Metadata(decBase):
    """Metadata SQLAlchemy Table"""
    __tablename__ = 'metadata'

    ID = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey('experiment.ID'))
    key = Column(Text(length=255))
    value = Column(MATLAB)

    experiment = relationship('Experiment', back_populates='meta')


class Entity(decBase):
    """Entity SQLAlchemy Table"""
    __tablename__ = 'entity'

    ID = Column(Integer, primary_key=True, autoincrement=True)
    sampleID = Column(Integer, ForeignKey('sample.ID'))

    objects = relationship('Object', back_populates='entity')
    sample = relationship('Sample', back_populates='entities')
    entityGroup = relationship('EntityGroup_Entities', back_populates='entities')


class Sample(decBase):
    __tablename__ = 'sample'

    ID = Column(Integer, primary_key=True, autoincrement=True)
    material = Column(String(50))
    supplier = Column(String(50))
    preparation_date = Column(DateTime)
    substrate = Column(String(50))
    preparation_method = Column(String(50))

    entities = relationship('Entity', back_populates='sample')


class Experiment(decBase):
    __tablename__ = 'experiment'

    ID = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String(50))

    meta = relationship('Metadata', back_populates='experiment')


class Measurement(decBase):
    __tablename__ = 'measurement'

    ID = Column(Integer, primary_key=True)
    experiment_ID = Column(Integer, ForeignKey('experiment.ID'), autoincrement=True)
    object_ID = Column(Integer, ForeignKey('object.ID'))
    data = Column(MATLAB)
    created = Column(DateTime, default=datetime.now())

    object = relationship('object', back_populates='measurement')


class EntityGroup(decBase):
    __tablename__ = 'entityGroup'

    ID = Column(Integer, primary_key=True)
    name = Column(String)
    details = Column(String)
    owner = Column(String)
    created = Column(DateTime)
    last_access = Column(DateTime)

    entities = relationship('EntityGroup_entity', back_populates='entityGroup')


class EntityGroupEntity(decBase):
    __tablename__ = 'entityGroup_entity'

    entityGroupID = Column(Integer, ForeignKey('entityGroup.ID'), primary_key=True)
    entityID = Column(Integer, ForeignKey('entity.ID'), primary_key=True)

    entityGroup = relationship('EntityGroup', back_populates='entities')
    entities = relationship('Entity', back_populates='entityGroup')


class Collections(decBase):
    """Collections"""
    __tablename__ = 'collections'
    collectionID = Column(String, primary_key=True)
    dbID = Column(Integer)
    created = Column(DateTime, default=datetime.now())
