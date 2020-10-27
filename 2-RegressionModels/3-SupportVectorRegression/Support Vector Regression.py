#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


dataset=pd.read_csv('Position_Salaries.csv')
df=pd.DataFrame(dataset) #Creating a dataframe
X=df.iloc[:,1:-1].values  
y=df.iloc[:,-1].values


# # Feature Scaling
# * Here we have to apply the feature scaling at dependant variable salaries also because we don't want the level feature to be neglected as compared to salary fetaure.
# * If the values is already between 0 to 1 than we don't need to apply the feature scaling proces.

# In[8]:


X  #Here it's a 2d array because the standard scalar expects the array to be 2d if it's 1d than it will throw an error


# In[9]:


y


# In[10]:


y=y.reshape(len(y),1)


# In[11]:


y #Here it's a 2d array because the standard scalar expects the array to be 2d if it's 1d than it will throw an error


# # Feature Scaling

# In[13]:


from sklearn.preprocessing import StandardScaler #standar values between -3 to +3
sc_X=StandardScaler()
sc_y=StandardScaler()  #here we have create the 2 varaiables because X have different mean and y have different mean so
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)


# In[14]:


X


# In[15]:


y


# # Train SVR Model on Whole Dataset

# In[16]:


from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)


# # Predict a New Result

# In[17]:


sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))


# # Visualizing the SVR Results

# In[19]:


plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color='red')
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X)), color='Blue')
plt.title('Truth or Bluff(Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel("Salary")
plt.show() 


# In[23]:


X_grid=np.arange(min(sc_X.inverse_transform(X)),max(sc_X.inverse_transform(X)),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color='red')
plt.plot(X_grid,sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color='Blue')
plt.title('Truth or Bluff(Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel("Salary")
plt.show()

