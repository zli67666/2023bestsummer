#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.datasets import load_wine


# In[2]:


data = load_wine()  # 數據
df = pd.DataFrame(data.data, columns=data.feature_names)
df['class'] = data.target


# In[3]:


df.head()


# In[4]:


mean_values = df.groupby('class').mean()
# 使用groupby 按照'class'列進行分組，並計算每個組的均值

print("三種酒 13個不同的化學特性特徵均值：")
mean_values


# In[5]:


# 建立不同類別的 DataFrame
class_0 = df[df['class'] == 0]
class_1 = df[df['class'] == 1]
class_2 = df[df['class'] == 2]


# In[6]:


print("三種酒p值：")

print("0_1：")
print(scipy.stats.ttest_ind(class_0,class_1) .pvalue)
print("1_2：")
print(scipy.stats.ttest_ind(class_1,class_2) .pvalue)
print("2_0：")
print(scipy.stats.ttest_ind(class_2,class_0) .pvalue)


# In[ ]:





# In[ ]:




