import pandas as pd
import numpy as np
import math

df = pd.read_csv('agaricus-lepiota.csv', header=None,
                 names=['class','cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stock-shape', 'stock-root', 'stock-surface-above-ring', 'stock-surface-below-ring', 'stock-color-above-ring','stock-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population','habitat'])

#print df.head(10)

del df['stock-root']

target = df['class']
#print target.as_matrix()
del df['class']

#print df.head()

data = df.as_matrix()


bruises = data[:,3]
bruises = data[:,3]
bruisesDict = {}
bruisesDict['t'] = 0
bruisesDict['f'] = 0
TDict = {'e':0, 'p':0}
FDict = {'e':0, 'p':0}
# print bruises
for i in range(len(bruises)):
    if bruises[i] == 't':
        bruisesDict['t'] += 1
        if target[i] == 'e':
            TDict['e'] += 1
        elif target[i] == 'p':
            TDict['p'] += 1
    elif bruises[i] == 'f':
        bruisesDict['f'] += 1
        if target[i] == 'e':
            FDict['e'] += 1
        elif target[i] == 'p':
            FDict['p'] += 1

print 'Bruises:', bruisesDict
print 'True, poision/eatable: ', TDict
print 'False, poision/eatble: ', FDict
print '% Bruised and Poison: ', (float(TDict['p'])/len(bruises))*100
print '% Bruised and Eatable: ', (float(TDict['e'])/len(bruises))*100
print '% Not Bruised and Poison: ', (float(FDict['p'])/len(bruises))*100
print '% Not Bruise and Eatable: ', (float(FDict['e'])/len(bruises))*100
print 'total %: ', (float(TDict['p'])/len(bruises))*100+(float(TDict['e'])/len(bruises))*100+(float(FDict['p'])/len(bruises))*100+(float(FDict['e'])/len(bruises))*100