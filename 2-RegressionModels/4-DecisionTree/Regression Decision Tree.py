#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[6]:


dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:-1].values  
y=dataset.iloc[:,-1].values
##iloc means locate index(rows,colums)
print("Features Data: \n ",X)
print("Target Data: \n ",y)


# # Notes
# * we can apply the data preprocessing steps based on the dataset
# * we don't need to apply feature scaling in regression decision trees because the predictions from the decision tree are from the different parts of the data unlike the other ones where the algorithm runs on the dataset something like sequential.
# * Decison tree model is not best used for single feature it's a best option for multiple features.
# 

# # Training The Decison Tree on Whole Dataset

# In[12]:


from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)


# # Prediction

# In[13]:


regressor.predict([[6.5]])


# # Visualizing the Results

# In[15]:


X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid), color='Blue')
plt.title('Truth or Bluff (Decision Tree)')
plt.xlabel('Position Level')
plt.ylabel("Salary")
plt.show()


# In[16]:


plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X), color='Blue')
plt.title('Truth or Bluff (Decision Tree)')  #Here it does not make sense because 
plt.xlabel('Position Level')
plt.ylabel("Salary")
plt.show()

