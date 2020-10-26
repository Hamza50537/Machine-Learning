#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


dataset=pd.read_csv('Position_Salaries.csv')
df=pd.DataFrame(dataset) #Creating a dataframe
X=df.iloc[:,1:-1].values  
y=df.iloc[:,-1].values


# In[4]:


X


# In[5]:


y


# # Training Linear Regression

# In[7]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)


# # Training Ploynomial Regression

# In[8]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)


# # Linear Regression Result

# In[9]:


plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X), color='Blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel("Salary")
plt.show()


# # Ploynomial Regression

# In[11]:


plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)), color='Blue')
plt.title('Truth or Bluff(Ploynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel("Salary")
plt.show()


# # Ploynomial Regression Results

# In[15]:


X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='Blue')
plt.title('Truth or Bluff(Ploynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel("Salary")
plt.show()


# # Prediction With Linear Regreesion

# In[18]:


regressor.predict([[7.5]])


# # Prediction With Polynomial Regreesion

# In[20]:


lin_reg_2.predict(poly_reg.fit_transform([[7.5]]))

