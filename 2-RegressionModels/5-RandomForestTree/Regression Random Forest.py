#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:-1].values  
y=dataset.iloc[:,-1].values
##iloc means locate index(rows,colums)
print("Features Data: \n ",X)
print("Target Data: \n ",y)


# # Notes
# * we can apply the data preprocessing steps based on the dataset
# * we don't need to apply feature scaling in regression random forest trees because the predictions from the random forest tree are from the different parts of the data unlike the other ones where the algorithm runs on the dataset something like sequential.
# * Decison tree model is not best used for single feature it's a best option for multiple features.
# 

# # Training the Random Forest Model 

# In[9]:


from sklearn.ensemble import RandomForestClassifier
regressor=RandomForestClassifier(n_estimators=10,random_state=0,)
regressor.fit(X,y)


# # Prediction

# In[10]:


regressor.predict([[6.5]])


# # Visualizing the Results

# In[11]:


X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid), color='Blue')
plt.title('Truth or Bluff (Random Forest Tree)')
plt.xlabel('Position Level')
plt.ylabel("Salary")
plt.show()

