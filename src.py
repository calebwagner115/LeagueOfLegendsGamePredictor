#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import AdaBoostClassifier
import statistics
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree


# In[2]:


def importFile(file):
    f = open(file,'r')
    entireFile = f.read()
    return entireFile


# In[3]:


games = importFile('games2.csv')[:-1]
champion_info = importFile('champion_info.json')
summoner_spell = importFile('summoner_spell_info.json')

#get rid of other formatting
champion_info = json.loads(champion_info)['data']
champion_spell = json.loads(summoner_spell)['data']


# In[4]:


testgames = games
games.join('')
games = games.split('\n')
for i in range(len(games)):
    test = games[i].join('')
    games[i] = games[i].split(',')


# In[5]:


allgames_df = pd.DataFrame(games)


# In[6]:



new_header = allgames_df.iloc[0] #grab the first row for the header
allgames_df = allgames_df[1:] #take the data less the header row
allgames_df.columns = new_header #set the header row as the df header

allgames_df = allgames_df[allgames_df.columns].applymap(np.int)
allgames_df.columns = new_header


# In[7]:



trainSamples_filtered = allgames_df.loc[:,'firstBlood'::]
trainActuals = allgames_df.loc[:,'winner':'winner':]

kNN_trainSamples_filtered = trainSamples_filtered.values.tolist()
kNN_trainActuals = trainActuals.values.tolist()
train_holdoutX, test_holdoutX, train_actualY, test_acutalY = train_test_split(kNN_trainSamples_filtered, kNN_trainActuals, test_size=0.7)


# In[8]:


print(len(train_holdoutX), len(test_holdoutX), len(train_actualY), len(test_acutalY))


# In[9]:


reg = LassoCV()
reg.fit(train_holdoutX, np.ravel(train_actualY))
coef = pd.Series(reg.coef_, index = trainSamples_filtered.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(train_holdoutX,train_actualY))
fitted = SelectKBest(chi2, k=7).fit(train_holdoutX, np.ravel(train_actualY))
trainSamples_filtered = fitted.transform(train_holdoutX)
testSamples_filtered = fitted.transform(test_holdoutX)


# In[10]:



test_train = allgames_df.loc[:,'firstBlood'::]

for bool, feature in zip(fitted.get_support(), test_train.columns.values):
    if bool:
        print(feature)


# In[11]:


test_acutalY[0]
# train_holdoutX[0]

neigh = KNeighborsClassifier(n_neighbors=7)
#neigh.fit(train_holdoutX, np.ravel(train_actualY))
neigh.fit(trainSamples_filtered, np.ravel(train_actualY))
pred=neigh.predict(testSamples_filtered)
ans = 0
for i in range(len(test_holdoutX)):
    if pred[i] == test_acutalY[i]:
        ans +=1;
print('kNN Score:\n',ans/i)
print(confusion_matrix(np.ravel(test_acutalY), pred))
val = cross_val_score(neigh, testSamples_filtered, np.ravel(test_acutalY), cv=5)
print(val, val.mean(), '\n')


# In[12]:



decision_tree = DecisionTreeClassifier(random_state=0, max_depth=6)
decision_tree = decision_tree.fit(trainSamples_filtered, np.ravel(train_actualY))
predictions = decision_tree.predict(testSamples_filtered)

correct=0
for i in range(len(predictions)):
    if predictions[i] == test_acutalY[i]:
        correct += 1
print('DecisionTree Score:\n',ans/i)
print('Percent correct: {}'.format(correct/i), len(np.ravel(test_acutalY)), len(predictions))

print(confusion_matrix(np.ravel(test_acutalY), predictions))


val = cross_val_score(decision_tree, train_holdoutX, np.ravel(train_actualY), cv=5)
print(val, val.mean(),'\n')


# In[99]:


clf = AdaBoostClassifier(n_estimators=5)
scaler = StandardScaler()
scaler.fit(trainSamples_filtered)


scaler.fit(trainSamples_filtered, np.ravel(train_actualY))
train = scaler.transform(trainSamples_filtered)

test = scaler.transform(testSamples_filtered)
scaler.fit(testSamples_filtered, np.ravel(train_actualY))

clf.fit(train,np.ravel(train_actualY))
predictions = clf.predict(testSamples_filtered)
correct=0
for i in range(len(predictions)):
    if predictions[i] == np.ravel(test_acutalY)[i]:
        correct += 1
print('AdaBoost Score:\n',ans/i)
print('Percent correct: {}'.format(correct/i))
print(confusion_matrix(test_acutalY, predictions), '\n')


# In[14]:


# Neural Network HERE
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(15,), max_iter=5000, random_state=1)
scaler = StandardScaler()

scaler.fit(trainSamples_filtered)


scaler.fit(trainSamples_filtered, np.ravel(train_actualY))
train = scaler.transform(trainSamples_filtered)

test = scaler.transform(testSamples_filtered)
scaler.fit(testSamples_filtered, np.ravel(train_actualY))

clf.fit(train,np.ravel(train_actualY))
# predictions = clf.predict(testSamples_filtered)
print('MLPClassifier Score:\n',ans/i)
vals = cross_val_score(clf, testSamples_filtered, np.ravel(test_acutalY), cv=5)
# vals.mean()
print(val, val.mean(),'\n')


# In[ ]:




