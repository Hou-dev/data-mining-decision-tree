#!/usr/bin/env python
# coding: utf-8

# In[179]:

import os
import numpy as np
import pandas as pd
from sklearn import tree, metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split 
import graphviz


data = pd.read_excel('cosc_GA2.xlsx', sep='|',names =['age','income','student','credit_rating','buys_computer'] , encoding='latin-1')
data.head()


# In[180]:


data['age'],class_names = pd.factorize(data['age'])
print(class_names)
print(data['age'].unique())


# In[181]:


data['income'],_ = pd.factorize(data['income'])
data['student'],_ = pd.factorize(data['student'])
data['credit_rating'],_ = pd.factorize(data['credit_rating'])
data['buys_computer'],_ = pd.factorize(data['buys_computer'])
data.head()


# In[182]:


data.info()


# In[199]:


x = data.iloc[:,:-1]
y = data.iloc[:,-1]


# In[200]:


x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=0.2, random_state=0)


# In[201]:


dtree = tree.DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)
dtree.fit(x_train,y_train)


# In[202]:


y_pred = dtree.predict(x_test)
count_misclassified = (y_test != y_pred).sum()
print('Misclassified sampled: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test,y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


# In[203]:


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

feature_names = x.columns
dot_data = tree.export_graphviz(dtree, out_file = None, filled=True, rounded=True,
                               feature_names=feature_names,
                               class_names=class_names)
graph = graphviz.Source(dot_data)
graph


# In[ ]:




