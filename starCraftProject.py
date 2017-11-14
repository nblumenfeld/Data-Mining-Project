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
import math
import graphviz



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
question1_sample = df[['ProtossPylon','ProtossSecondPylon','ProtossFirstGas','ProtossSecondGas','ProtossFirstExpansion','ProtossGateway','ProtossGroundWeapons1','ProtossShields1','ProtossCitadel','midBuild']]
df = df.iloc[:,:-1]
data = df.as_matrix()
name = df.columns.values


# build_data = df.sample(frac=.8)
#
# test_data = df.loc[~df.index.isin(build_data.index)]

stds = 1.0  # Number of standard deviation that defines 'outlier'.
z = question1_sample[['ProtossPylon','ProtossSecondPylon','ProtossFirstGas','ProtossSecondGas','ProtossFirstExpansion','ProtossGateway','ProtossGroundWeapons1','ProtossShields1','ProtossCitadel','midBuild']].groupby('midBuild').transform(
    lambda group: (group - group.mean()).div(group.std()))
outliers = z.abs() > stds

data = question1_sample[outliers.any(axis=1)]

build_data = data.sample(frac=.8)

test_data = data.loc[~data.index.isin(build_data.index)]

#print outliers



build_data_labels = build_data['midBuild']
test_data_labels = test_data['midBuild']
build_data = build_data.iloc[:,:-1]
test_data = test_data.iloc[:,:-1]
data = build_data.as_matrix()
test = test_data.as_matrix()
name = df.columns.values

clf = tree.DecisionTreeClassifier()
clf = clf.fit(data,build_data_labels)
print clf.predict(test)


sc_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(sc_data)
graph.render("StarCraft")


#print df.head(5)
#print target