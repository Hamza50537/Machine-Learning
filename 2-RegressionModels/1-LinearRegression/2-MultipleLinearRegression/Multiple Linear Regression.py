#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[45]:


dataset=pd.read_csv('50_Startups.csv')
df=pd.DataFrame(dataset) #Creating a dataframe
X=df.iloc[:,:-1].values  
y=df.iloc[:,-1].values


# In[46]:


df.head()


# In[47]:


df.info()


# In[48]:


df.describe()


# # Encoding the Categorical Data
# 
# We have seen from the data that we have a feature with the name of <b>"State"</b> which has differnet countries names and it's like a classification so first we will convert it into more optimal form.

# In[49]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoding',OneHotEncoder(),[3])],remainder='passthrough')
X=np.array(ct.fit_transform(X))


# In[50]:


print(X)


# # Feature Scaling
# Here we don't need to apply feature scaling because we know that multiple regression is the same as simple linear regression but with some extra features and constants such as y=b0+b1x1+b2x2. coefficients will deal with the feature dominance

# # Split the Data Into Train and Test

# In[51]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state = 0)


# In[55]:


print(y_train)


# # Training Model on Training Set

# In[52]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# # Predicting the Test Results

# In[57]:


Y_pred=regressor.predict(X_test)
np.set_printoptions(precision=2) #display any numerical value after 2 decimal
# 1-Here we are concatinating the trained salary and predicted salary because we can't plot the 5d data.
# 2-We also applied the reshape(rows,cols) function to show the data in vertical form rather than horizontal form.
# 3-axis=0 means vertical concatination. axis=1 means horizontal concatination.
# we have done axis=1 because origional data is in horizontal form we have reshaped it in the print statement
print(np.concatenate((Y_pred.reshape(len(Y_pred),1),y_test.reshape(len(y_test),1)),axis=1))  

