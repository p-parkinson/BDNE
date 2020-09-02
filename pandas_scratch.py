# Implement data analysis with sqlalchemy -> pandas/numpy -> matplotlib stack

# Imports - db_orm for db connection, pandas for dataframe, numpy for manupulation, matplotlib for plotting
from db_orm import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample to study
sampleUnderStudy = 25

# SQL query - select sample spectra, position, and pic
sql = session.query(Measurement.ID.label('measurementID'), Sample.material, Entity.ID.label('EntityID'),
                    Experiment.type, Measurement.data). \
    join(Experiment). \
    join(Object). \
    join(Entity). \
    join(Sample). \
    filter(Sample.ID == sampleUnderStudy). \
    filter(Experiment.type.in_(['spectra', 'pic', 'position'])). \
    limit(500)

# Pull into dataFrame
dataFrame = pd.read_sql_query(sql.statement, session.bind, index_col='measurementID')
print('Retrieved {} experiments'.format(len(dataFrame.index)))
# Note data type is categorical - convert as such
dataFrame["catType"] = dataFrame["type"].astype('category')
# Find spectral maximum for the spectra
dataFrame["specMax"] = dataFrame["data"][dataFrame['catType'] == 'spectra'].apply(np.max)
# Find image size for pics
dataFrame["picSize"] = dataFrame["data"][dataFrame['catType'] == 'pic'].apply(np.shape)

# Cut incorrect image sizes - keep only those with more than 10% of images
# sizeCount = dataFrame.picSize.value_counts()
# picSizes = pd.DataFrame(data={'picSize': sizeCount.index, 'count': sizeCount})
# picSizes['norm'] = picSizes.picSize.apply(np.linalg.norm)
# picSizes['fracCount'] = picSizes["count"] / picSizes["count"].sum()
# to_use = picSizes['picSize'][picSizes['fracCount'] > 0.1]

# Split & apply cuts
info = dataFrame[{'EntityID', 'material'}].groupby(['EntityID']).first()
info['material'] = info['material'].astype('category')

pics = dataFrame[{'EntityID', 'data', 'picSize'}].rename(columns={"data": "pic"})

spec = dataFrame[{'EntityID', 'data', 'specMax'}][dataFrame['specMax'] > 500].rename(columns={"data": "spec"})

posn = dataFrame[{'EntityID', 'data'}][dataFrame['catType'] == 'position'].rename(columns={"data": "position"})
posn[['X', 'Y']] = pd.DataFrame(posn['position'].to_list(), index=posn.index)

# Combine and drop where missing
data = info.merge(pics, on='EntityID').merge(spec, on='EntityID').merge(posn, on='EntityID').dropna()
print('Cut to {} entities'.format(len(data.index)))

# To show - central data and 8 surrounding images.
# Select
showset = data.sample(n=8)
sel = data.EntityID.isin(showset.EntityID).to_numpy()
# Plot
plt.clf()
plt.subplot(3, 3, 5)
plt.scatter(data['X'], data['Y'], s=0.5 + sel * 2, c=sel)
plt.axis('image')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
surr = [1, 2, 3, 4, 6, 7, 8, 9]
for (i, w) in zip(surr, showset.iterrows()):
    plt.subplot(3, 3, i)
    plt.imshow(w[1]['pic'])
    plt.title('EntID: {}'.format(w[1]['EntityID']))
    plt.axis('image')
    plt.xticks([])
    plt.yticks([])
plt.show()
