'''
Data Mining Project fall 2017
Step2

Thomas Mannsverk Elissen
Bridger Fisher
Ian Sime
Noah Blumenfeld

'''

import pandas as pd
import numpy as np
from sklearn import tree
import math

#read the csv file but exclude the first row
#1140 rows
#53 attributes

df = pd.read_csv(r'https://github.com/bgweber/StarCraftMining/raw/master/data/scmPvT_Protoss_Mid.csv', skiprows=1, header=None, names = ['ProtossPylon','ProtossSecondPylon','ProtossFirstGas','ProtossSecondGas','ProtossFirstExpansion','ProtossSecondExpansion','ProtossThirdExpansion','ProtossFourthExpansion','ProtossGateway','ProtossSecondGatway','ProtossThirdGatway','ProtossFourthGatway','ProtossCore','ProtossZealot','ProtossGoon','ProtossRange','ProtossForge',
'ProtossCannon','ProtossGroundWeapons1','ProtossGroundArmor1','ProtossShields1','ProtossGroundWeapons2','ProtossGroundArmor2','ProtossShields2','ProtossCitadel','ProtossLegs','ProtossArchives','ProtossTemplar','ProtossArchon','ProtossStorm','ProtossDarkTemplar','ProtossDarkArchon','ProtossMaelstorm','ProtossRoboBay','ProtossShuttle','ProtossShuttleSpeed','ProtossRoboSupport','ProtossReavor','ProtossReavorDamage','ProtossReavorCapacity','ProtossObservory','ProtossObs',
'ProtossStargate','ProtossCorsair','ProtossDisruptionWeb','ProtossFleetBeason','ProtossCarrier','ProtossCarrierCapacity','ProtossTribunal',
'ProtossArbitor','ProtossStatis','ProtossRecall','ProtossAirWeapons1','ProtossAirArmor1','ProtossAirWeapons2','ProtossAirArmor2','midBuild'])


target = df['midBuild'].as_matrix()
#df = df.iloc[:,:-1]
dataMatrix = df.as_matrix()
name = df.columns.values


stds = 2.0  # Number of standard deviation that defines 'outlier'.
z = df.groupby('midBuild').transform(lambda group: (group - group.mean()).div(group.std()))
outliers = z.abs() > stds

data =df[outliers.any(axis=1)]
build_data = data.sample(frac=.8)
test_data = data.loc[~data.index.isin(build_data.index)]

#######
#from step1
build_data_labels = build_data['midBuild']
test_data_labels = test_data['midBuild']
build_data = build_data.iloc[:,:-1]
test_data = test_data.iloc[:,:-1]
data = build_data.as_matrix()
test = test_data.as_matrix()
name = df.columns.values

clf = tree.DecisionTreeClassifier()
clf = clf.fit(data,build_data_labels)
prediction = clf.predict(test)

midBuild_set = set(test_data_labels)


classif_rate = np.mean(prediction.ravel() == test_data_labels.ravel()) * 100
print("classif_rate for %s : %f " % ('DecisionTree', classif_rate))

###########
#DT
#kNN
#Forest
#GradientBoost
#Regression
#Kmeans clustering
