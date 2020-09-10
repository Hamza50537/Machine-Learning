#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[26]:


dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values  
y=dataset.iloc[:,-1].values
##iloc means locate index(rows,colums)
print("Features Data: \n ",X)
print("Target Data: \n ",y)


# **Encoding: Here we will apply the encoding scheme on the country colum(Categorical Data) because we have repition of countries names and we will calculate the result multiple times for the same country. So that's why we will represent them with One Hot encoding scheme.

# <b>Encoding the independant variable

# In[27]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoding',OneHotEncoder(),[0])],remainder='passthrough')
X=np.array(ct.fit_transform(X))
print(X)


# <b>Encoding the dependant variable

# In[28]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)


# <b>Splitting the data into train and test

# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[30]:


print("X_train:\n ",X_train)


# In[31]:


print("y_train:\n ",y_train)


# In[32]:


print("X_test:\n ",X_test)


# In[33]:


print("y_test:\n ",y_test)


# **Feature Scaling is being done so to eliminate the dominance of 1 particular fetaure so that all features will gets the equal chance. It depends on your dataset and it don't necessarily needed all the time.
# There are 2 ways to do feature scaling:
# 
# **1-Standarisation: 
# s=[Xi-mean(x)]/[standard deviation(x)] {All features value will be between -3,+3}
# 
# Explain: Here we will subtract the each value of the feature by the mean of that feature and then divide it with the standard deviation of that feature.
# 
# **2-Normalisation:
# n=[Xi-min(x)]/[max(x)-min(x)] {All features value will be between 0,1}
# 
# Explain: Here we will subtract the each value of the feature by the minimum value of that feature and then divide it with the max(feature)-min(feature) of that feature.
# 
# **Standarisation will work all the time and Normalisation works good on the normalized distributions so standarisation is prefferd over normalisation.

# <b>Feature Scaling

# In[34]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train[:,3:]=sc.fit_transform(X_train[:,3:])
X_test[:,3:]=sc.transform(X_test[:,3:])


# In[35]:


print(X_train)


# In[36]:


print(X_test)

