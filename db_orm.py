# SQLAlchemy-based ORM model for the BDNE project
# Author : Patrick Parkinson <patrick.parkinson@manchester.ac.uk>
# Date : 30/06/2020

from sqlalchemy.orm import Session, relationship
from sqlalchemy import create_engine, Integer, ForeignKey, Text, String, Column, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy.types as types
import configparser
import os
import zlib
from datetime import datetime
from matlab_serialise import serialise, deserialise

# Get current path (required for locating config.ini)
package_directory = os.path.dirname(os.path.abspath(__file__))

# Read and import configuration from config file
config = configparser.ConfigParser()
config.read(os.path.join(package_directory, 'config.ini'))

# Create base autoloader
decBase = declarative_base()
# Connect to the database using credentials from config.ini
engine = create_engine('mysql+mysqlconnector://%s:%s@%s:%s/bdne' % (
    config.get('DATABASE', 'user'), config.get('DATABASE', 'pass'),
    config.get('DATABASE', 'server'), config.get('DATABASE', 'port')))


# Set up serialise/deserialise
class MATLAB(types.TypeDecorator):
    """Converts to and from hlp_serialised MATLAB form on the fly"""
    impl = types.LargeBinary

    def process_bind_param(self, value, dialect):
        return serialise(value)

    def process_result_value(self, value, dialect):
        if value[0] == 201:
            # Dataset is compressed. Cut first value, decompress with zlib
            value = zlib.decompress(value[1:])
        return deserialise(value)


# Set up the tables
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
    value = Column(Text(length=65535))

    experiment = relationship('Experiment', back_populates='meta')


class Entity(decBase):
    """Entity SQLAlchemy Table"""
    __tablename__ = 'entity'

    ID = Column(Integer, primary_key=True, autoincrement=True)
    sampleID = Column(Integer, ForeignKey('sample.ID'))

    objects = relationship('Object', back_populates='entity')
    sample = relationship('Sample', back_populates='entities')


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


# Set up a session
session = Session(engine)
