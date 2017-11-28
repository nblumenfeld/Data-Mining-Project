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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import preprocessing
import plotly.plotly as py
import plotly.graph_objs as go
import math
from sklearn.model_selection import GridSearchCV

#read the csv file but exclude the first row
#1140 rows
#53 attributes

df = pd.read_csv(r'https://github.com/bgweber/StarCraftMining/raw/master/data/scmPvT_Protoss_Mid.csv', skiprows=1, header=None, names = ['ProtossPylon','ProtossSecondPylon','ProtossFirstGas','ProtossSecondGas','ProtossFirstExpansion','ProtossSecondExpansion','ProtossThirdExpansion','ProtossFourthExpansion','ProtossGateway','ProtossSecondGatway','ProtossThirdGatway','ProtossFourthGatway','ProtossCore','ProtossZealot','ProtossGoon','ProtossRange','ProtossForge',
'ProtossCannon','ProtossGroundWeapons1','ProtossGroundArmor1','ProtossShields1','ProtossGroundWeapons2','ProtossGroundArmor2','ProtossShields2','ProtossCitadel','ProtossLegs','ProtossArchives','ProtossTemplar','ProtossArchon','ProtossStorm','ProtossDarkTemplar','ProtossDarkArchon','ProtossMaelstorm','ProtossRoboBay','ProtossShuttle','ProtossShuttleSpeed','ProtossRoboSupport','ProtossReavor','ProtossReavorDamage','ProtossReavorCapacity','ProtossObservory','ProtossObs',
'ProtossStargate','ProtossCorsair','ProtossDisruptionWeb','ProtossFleetBeason','ProtossCarrier','ProtossCarrierCapacity','ProtossTribunal',
'ProtossArbitor','ProtossStatis','ProtossRecall','ProtossAirWeapons1','ProtossAirArmor1','ProtossAirWeapons2','ProtossAirArmor2','midBuild'])


#df = df.iloc[:,:-1]
data = df.as_matrix()
name = df.columns.values

##########################################################
'''
#Our own Decision DecisionTree
print "Our own Decision, with a STDS of 2.0"
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
print

###########################################################
'''

'''
All code below relates to step 2 of the project
'''

for col in range(1,56):
    m=df.iloc[:,col].dropna().quantile(0.99)
    df.iloc[:,col]=df.iloc[:,col].map(lambda x: None if x>m else x)


df =df.dropna()
#Build and test data
build_data = df.sample(frac=.8)
test_data = df.loc[~df.index.isin(build_data.index)]
build_data_labels = build_data['midBuild']
test_data_labels = test_data['midBuild']
build_data = build_data.iloc[:,:-1]
test_data = test_data.iloc[:,:-1]

target = df['midBuild'].as_matrix()
df = df.iloc[:,:-1]
df = preprocessing.StandardScaler().fit_transform(df)


# findBestscore = []
# '''
# Decision tree
# '''
# print "Decision tree"
# model_DT = DecisionTreeClassifier()
# tuned_parameters = {'criterion':["gini","entropy"]}
# model_DT.fit(df,target)
# model_DT_GSCV = GridSearchCV(model_DT,tuned_parameters,cv=5,verbose=1)
# model_DT_GSCV.fit(df,target)
# print model_DT_GSCV.best_params_
# print model_DT_GSCV.best_score_
# print
# findBestscore.append(("Decision Tree", model_DT_GSCV.best_score_))

# '''
# Random Forest
# '''
# print "Random Forest"
# model_RF = RandomForestClassifier()
# tuned_parameters = {'n_estimators':[10,20], 'max_depth':[None, 3]}
# model_RF.fit(df,target)
# model_RF_GSCV = GridSearchCV(model_RF,tuned_parameters,cv=3,verbose=1)
# model_RF_GSCV.fit(df,target)
# print model_RF_GSCV.best_params_
# print model_RF_GSCV.best_score_
# print
# findBestscore.append(("Random Forest", model_RF_GSCV.best_score_))


# '''
# Gradient Booster
# '''
# print "Gradient Booster"
# model_GB = GradientBoostingClassifier()
# tuned_parameters = {'n_estimators':[100,50], 'max_depth':[2, 3]}
# model_GB.fit(df,target)
# model_GB_GSCV = GridSearchCV(model_GB,tuned_parameters,cv=3,verbose=1)
# model_GB_GSCV.fit(df,target)
# print model_GB_GSCV.best_params_
# print model_GB_GSCV.best_score_
# print
# findBestscore.append(("Gradient Booster", model_GB_GSCV.best_score_))


# '''
# KNN
# '''
# print "KNN"
# model_KNN = KNeighborsClassifier()
# tuned_parameters = {'n_neighbors':[5,9,15],'weights':['uniform','distance']}
# model_KNN.fit(df,target)
# model_KNN_GSCV = GridSearchCV(model_KNN, tuned_parameters, cv=3 ,verbose=1)
# model_KNN_GSCV.fit(df,target)
# print model_KNN_GSCV.best_params_
# print model_KNN_GSCV.best_score_
# print
# findBestscore.append(("KNN", model_KNN_GSCV.best_score_))


# '''
# Logistic Regression
# '''
# print "Logistic Regression"
# model_LR = LogisticRegression()
# tuned_parameters = {'penalty':['l1','l2']}
# model_LR.fit(df,target)
# model_LR_GSCV = GridSearchCV(model_LR,tuned_parameters, cv=5, verbose=1)
# model_LR_GSCV.fit(df,target)
# print model_LR_GSCV.best_params_
# print model_LR_GSCV.best_score_
# print
# findBestscore.append(("Logistic Regression", model_LR_GSCV.best_score_))

# from operator import itemgetter
# print findBestscore
# print max(findBestscore,key=itemgetter(1))
'''
Visualization
Pre Processing
Normalize
Evaluation
Plot Feature Importance
'''

model_best = GradientBoostingClassifier(n_estimators=50,max_depth=2)
model_best.fit(build_data,build_data_labels)
yhat = model_best.predict(build_data)
value = model_best.feature_importances_


# model_DT.fit(build_data, build_data_labels)
# yhat=model_DT.predict(build_data)

# value = model_DT.feature_importances_

# print value

ind=sorted(range(len(value)),reverse=False,key=lambda k: value[k])
#print ind
features=name[ind]
# print features
value=sorted(value,reverse=True)
value = value[:10]
# print value
ind=np.array(range(10))
# print ind
plt.rcParams['figure.figsize'] = (9,7)
plt.barh(bottom=ind,height=0.5,width=value,color='r')
plt.yticks(ind+0.25,features)
plt.xlabel('Weights')
plt.ylabel('Features')
plt.title('Feature Importances')
#plt.subplots_adjust(left=0.2)
plt.tight_layout()
#plt.savefig('feature_importances.png', format='png', dpi=300)
plt.show()

trace = go.Pie(labels=features, values=value)

py.iplot([trace], filename='GB_pie_chart')