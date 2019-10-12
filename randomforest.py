#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[29]:


df_train = pd.read_csv(r"C:\Users\h\Desktop\AI_Model\kaggle\house-prices-advanced-regression-techniques\train.csv")


# In[30]:


df_test = pd.read_csv(r"C:\Users\h\Desktop\AI_Model\kaggle\house-prices-advanced-regression-techniques\test.csv")


# In[31]:


#checking null values
sns.heatmap(df_train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[32]:


print(df_train.shape)


# In[33]:


#printing the NULL VALUES AND details of coloumns
for column in df_train:
    if df_train[column].isnull().any():
       print('{0} has {1} null values and type: {2}'.format(column, df_train[column].isnull().sum(), df_train[column].dtype))


# In[34]:


#dropping the column having majority of null values
df_train.drop(['MiscFeature'],axis=1,inplace=True) 
df_train.drop(['PoolQC'],axis=1,inplace=True) 
df_train.drop(['Fence'],axis=1,inplace=True) 
df_train.drop(['Alley'],axis=1,inplace=True) 
df_train.drop(['Id'],axis=1,inplace=True) 
#same things for test datasets
df_test.drop(['MiscFeature'],axis=1,inplace=True) 
df_test.drop(['PoolQC'],axis=1,inplace=True) 
df_test.drop(['Fence'],axis=1,inplace=True) 
df_test.drop(['Alley'],axis=1,inplace=True) 
df_test.drop(['Id'],axis=1,inplace=True)


# In[35]:


#Remaining null values filling with mean and mode(max frequent daTA)

for column in df_train:
    if df_train[column].isnull().any():
       print('{0} has {1} null values and type: {2}'.format(column, df_train[column].isnull().sum(), df_train[column].dtype))
       if df_train[column].dtype == 'object':
            df_train[column] = df_train[column].fillna(df_train[column].mode()[0])
       else:
            df_train[column] = df_train[column].fillna(df_train[column].mean())


# In[36]:


#checking null values
sns.heatmap(df_train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[37]:


#For test dataset printing the NULL VALUES AND details of coloumns
for column in df_test:
    if df_test[column].isnull().any():
       print('{0} has {1} null values and type: {2}'.format(column, df_test[column].isnull().sum(), df_test[column].dtype))


# In[38]:


for column in df_test:
    if df_test[column].isnull().any():
       print('{0} has {1} null values and type: {2}'.format(column, df_test[column].isnull().sum(), df_test[column].dtype))
       if df_test[column].dtype == 'object':
            df_test[column] = df_test[column].fillna(df_test[column].mode()[0])
       else:
            df_test[column] = df_test[column].fillna(df_test[column].mean())


# In[39]:


#checking null values
sns.heatmap(df_test.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[40]:


#Preserving y_train variable and drping from train datadets
y_train = df_train['SalePrice'].values
df_train.drop(['SalePrice'],axis=1,inplace=True) 
print(y_train.shape)


# In[41]:


#validating the shape fot train and test datasets
print(df_train.shape, df_test.shape, y_train.shape)


# In[42]:


#After observing the trainee and test datasets i found that both having diffrent [column].value_counts() ie. diffrent no. fo coloumn will be formed after one hot encoding
conct_df = pd.concat([df_train, df_test], axis=0)
print(conct_df.shape)


# In[43]:


#One hot encoding on concatinated df
data = pd.get_dummies(conct_df)
print(data.shape)


# In[44]:


x_train, x_test = data.iloc[:1460,:].values, data.iloc[1460:,:].values


# In[45]:


print(x_train.shape, df_train.shape)
print(x_test.shape, df_test.shape)


# In[46]:


#for training data x_ train spliting training and test sets
from sklearn.model_selection import train_test_split
xt_train, xt_test, yt_train, yt_test = train_test_split(x_train, y_train, test_size=0.1, random_state = 0)


# In[63]:


#finding the best n_estimators and random_state
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=5)
regressor.fit(xt_train, yt_train)


# In[64]:


#model parameter study
from sklearn.metrics import mean_squared_error, r2_score
model_score = regressor.score(xt_train, yt_train)
print("Coeffit. of determination of R^2: ",model_score)
y_predict = regressor.predict(xt_test)
print(" Mean squared error: ", mean_squared_error(yt_test, y_predict))
print("Test variance score:", r2_score(yt_test, y_predict))


# In[65]:


#Finally we are fitting whole train data without any split for submission file
model = RandomForestRegressor(n_estimators=100, random_state=5)
model .fit(x_train, y_train)


# In[66]:


y_test = model.predict(x_test)


# In[67]:


#Making the submission file
pred = pd.DataFrame(y_test)
sample_df = pd.read_csv(r'C:\Users\h\Desktop\AI_Model\kaggle\house-prices-advanced-regression-techniques\sample_submission.csv')
final_df = pd.concat([sample_df['Id'], pred], axis=1)
final_df.columns = ['Id', 'SalePrice']
final_df.to_csv(r'C:\Users\h\Desktop\AI_Model\kaggle\house-prices-advanced-regression-techniques\sample_submission.csv', index=False)


# In[ ]:




