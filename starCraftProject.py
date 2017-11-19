'''
Data Mining Project fall 2017

Thomas Mannsverk Elissen
Bridger Fisher
Ian Sime
Noah Blumenfeld

'''


import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import preprocessing
import math
# import graphviz



#read the csv file but exclude the first row
#1140 rows
#53 attributes

#912 for build data if using full 1140, 228 for test
#select 12 attributes at a time to build tree off of

#obviously we need to change this after we trim out rows for outliers and missing/redundant data

df = pd.read_csv(r'https://github.com/bgweber/StarCraftMining/raw/master/data/scmPvT_Protoss_Mid.csv', skiprows=1, header=None, names = ['ProtossPylon','ProtossSecondPylon','ProtossFirstGas','ProtossSecondGas','ProtossFirstExpansion','ProtossSecondExpansion','ProtossThirdExpansion','ProtossFourthExpansion','ProtossGateway','ProtossSecondGatway','ProtossThirdGatway','ProtossFourthGatway','ProtossCore','ProtossZealot','ProtossGoon','ProtossRange','ProtossForge',
'ProtossCannon','ProtossGroundWeapons1','ProtossGroundArmor1','ProtossShields1','ProtossGroundWeapons2','ProtossGroundArmor2','ProtossShields2','ProtossCitadel','ProtossLegs','ProtossArchives','ProtossTemplar','ProtossArchon','ProtossStorm','ProtossDarkTemplar','ProtossDarkArchon','ProtossMaelstorm','ProtossRoboBay','ProtossShuttle','ProtossShuttleSpeed','ProtossRoboSupport','ProtossReavor','ProtossReavorDamage','ProtossReavorCapacity','ProtossObservory','ProtossObs','ProtossStargate','ProtossCorsair','ProtossDisruptionWeb','ProtossFleetBeason','ProtossCarrier','ProtossCarrierCapacity','ProtossTribunal',
'ProtossArbitor','ProtossStatis','ProtossRecall','ProtossAirWeapons1','ProtossAirArmor1','ProtossAirWeapons2','ProtossAirArmor2','midBuild'])

target = df['midBuild'].as_matrix()
df = df.iloc[:,:-1]
data = df.as_matrix()
name = df.columns.values


'''
All commented code below was what we did for step 1 of the project
'''

# question1_sample = df[['ProtossPylon','ProtossSecondPylon','ProtossFirstGas','ProtossSecondGas','ProtossFirstExpansion','ProtossGateway','ProtossGroundWeapons1','ProtossShields1','ProtossCitadel','midBuild']]
# build_data = df.sample(frac=.8)
# test_data = df.loc[~df.index.isin(build_data.index)]

'''
 The block of code below looks at our sample data and then based on the number of standard deviations we specifiy, in this case 1, will trim out outlier rows.
 Since we chose 1 standard deviation as our cutoff, all the remaining data in the data set represents the average 95% of the set (Thomas check this number before you add it to the paper, I'm just going off what I remember from stats.)
 Somehow the grouping is done be the labels that are associated with each row but I'm pretty sure it's looking at the other 9 columns as it's basis for what is an outlier or not.
'''

# stds = 1.0  # Number of standard deviation that defines 'outlier'.
# z = question1_sample[['ProtossPylon','ProtossSecondPylon','ProtossFirstGas','ProtossSecondGas','ProtossFirstExpansion','ProtossGateway','ProtossGroundWeapons1','ProtossShields1','ProtossCitadel','midBuild']].groupby('midBuild').transform(
#     lambda group: (group - group.mean()).div(group.std()))
# outliers = z.abs() > stds
# data = question1_sample[outliers.any(axis=1)]
# build_data = data.sample(frac=.8)
# test_data = data.loc[~data.index.isin(build_data.index)]
# build_data_labels = build_data['midBuild']
# test_data_labels = test_data['midBuild']
# build_data = build_data.iloc[:,:-1]
# test_data = test_data.iloc[:,:-1]
# data = build_data.as_matrix()
# test = test_data.as_matrix()
# name = df.columns.values

# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(data,build_data_labels)
# prediction = clf.predict(test)

# midBuild_set = set(test_data_labels)

# sc_data = tree.export_graphviz(clf, out_file=None, feature_names=['ProtossPylon','ProtossSecondPylon','ProtossFirstGas','ProtossSecondGas','ProtossFirstExpansion','ProtossGateway','ProtossGroundWeapons1','ProtossShields1','ProtossCitadel'], class_names=['fastObs','FastDT', 'ReaverDrop', 'FastExpand', 'FastLegs', 'Carrier', 'Unknown'], filled=True, rounded=True, special_characters=True)
# graph = graphviz.Source(sc_data)
# graph.render("StarCraft")

# classif_rate = np.mean(prediction.ravel() == test_data_labels.ravel()) * 100
# print("classif_rate for %s : %f " % ('DecisionTree', classif_rate))



'''
All code below relates to step 2 of the project
'''
normalized_data_set = preprocessing.StandardScaler().fit_transform(data)


