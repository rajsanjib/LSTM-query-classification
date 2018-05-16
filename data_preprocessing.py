import pandas as pd
import sklearn 
import numpy as np

ds1 = pd.read_csv('labeler1.txt', sep='\t', lineterminator='\r')
ds2 = pd.read_csv('labeler2.txt', sep='\t', lineterminator='\r')

ds = pd.concat([ds1, ds2])
df_X = ds.iloc[:,0].values
X = pd.DataFrame(df_X)
df_y = ds.iloc[:,1:].values
y = pd.DataFrame(df_y)


# Taking care of missing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', axis = 0)
imputer = imputer.fit(y[:, :])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Decoding categorical data
