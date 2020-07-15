# Implement data analysis with sqlalchemy -> pandas/numpy -> matplotlib stack

from db_orm import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SQL query - select sample 2 spectra and pic
sql = session.query(Measurement.ID.label('measurementID'), Sample.material, Entity.ID.label('EntityID'),
                    Experiment.type, Measurement.data). \
    join(Experiment). \
    join(Object). \
    join(Entity). \
    join(Sample). \
    filter(Sample.ID == 2).filter(Experiment.type.in_(['spectra', 'pic'])). \
    limit(2500)

# Pull into dataFrame
dataFrame = pd.read_sql_query(sql.statement, session.bind, index_col='measurementID')
# Note data type is categorical - convert as such
dataFrame["catType"] = dataFrame["type"].astype('category')
# Find spectral maximum for the spectra
dataFrame["specMax"] = dataFrame["data"][dataFrame['catType'] == 'spectra'].apply(np.max)
# Find image size for pics
dataFrame["picSize"] = dataFrame["data"][dataFrame['catType'] == 'pic'].apply(np.shape)

# Cut incorrect image sizes - keep only those with more than 10% of images
sizeCount = dataFrame.picSize.value_counts()
picSizes = pd.DataFrame(data={'picSize': sizeCount.index, 'count': sizeCount})
picSizes['norm'] = picSizes.picSize.apply(np.linalg.norm)
picSizes['fracCount'] = picSizes["count"]/picSizes["count"].sum()
to_use = picSizes['picSize'][picSizes['fracCount'] > 0.1]
# TODO: Complete cut!
