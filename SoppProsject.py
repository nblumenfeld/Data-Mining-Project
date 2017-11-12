import pandas as pd
import numpy as np
import math

df = pd.read_csv('agaricus-lepiota.csv', header=None,
                 names=['cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stock-shape', 'stock-root', 'stock-surface-above-ring', 'stock-surface-below-ring', 'stock-color-above-ring','stock-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population','habitat'])

print df.head(10)