# Testing code for accessing data in AWS database

import numpy as np
from matlab_serialise import serialise, deserialise
import matplotlib.pyplot as plt
from db_orm import *


def fetch_random(all_measurements: np.ndarray, n: int) -> np.ndarray:
    # Fetch a set of n measurement data from the database
    tofetch = np.random.randint(1, len(all_measurements), n)
    selectedmeasurementids = [i.item() for i in all_measurements[tofetch]]
    # New sql query to collect measurements from the data
    getexp = session.query(Object.ID, Measurement.data). \
        join(Experiment).join(Object). \
        filter(Measurement.ID.in_(selectedmeasurementids))
    return getexp.all()


def show_images(pic_data: np.ndarray):
    # iterate over each image and show it
    for i in range(len(pic_data)):
        data = deserialise(pic_data[i][1])
        plt.subplot(3, 4, i + 1)
        plt.imshow(data)
        plt.xticks([])
        plt.yticks([])
        plt.axis('image')
        plt.title(pic_data[i][0])
    plt.show()
    plt.close()


def show_spectra(spectra_data: np.ndarray):
    # iterate over each spectra and show it
    for i in range(len(spectra_data)):
        data = deserialise(spectra_data[i][1])
        plt.subplot(3, 4, i + 1)
        plt.plot(data)
        plt.yticks([])
        plt.title(spectra_data[i][0])
    plt.show()
    plt.close()


####################################################################

displayType = 'spectra'

# Create sql select
stm = session.query(Measurement.ID).join(Experiment). \
    filter(Experiment.type == displayType)

all_pics = stm.all()
# Randomly select n
all_pics = np.array([i[0] for i in all_pics])
dr = fetch_random(all_pics, 12)

if displayType == 'pic':
    show_images(dr)
elif displayType == 'spectra':
    show_spectra(dr)
