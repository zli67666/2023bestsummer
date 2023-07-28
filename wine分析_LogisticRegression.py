#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_wine


# In[2]:


data = load_wine()


# In[3]:


print(type(data))


# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix 


# In[5]:


df = pd.DataFrame(data.data, columns=data.feature_names)


# In[6]:


target = data.target


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)


# In[8]:


model = LogisticRegression()


# In[9]:


model.fit(X_train, y_train)


# In[10]:


y_pred = model.predict(X_test)


# In[11]:


cm = confusion_matrix(y_test, y_pred)


# In[12]:


print("Confusion Matrix:")
print(cm)


# In[13]:


accuracy = (cm[0, 0] + cm[1, 1] + cm[2, 2]) / cm.sum()
print("Accuracy:", accuracy)


# In[ ]:




