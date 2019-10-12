#!/usr/bin/env python

import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train.shape, test.shape
train.describe()
train.head()
train.columns
train.info()
def missing_percentage(df):
    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])


def percent_value_counts(df, feature):
    """This function takes in a dataframe and a column and finds the percentage of the value_counts"""
    percent = pd.DataFrame(round(df.loc[:,feature].value_counts(dropna=False, normalize=True)*100,2))
    ## creating a df with th
    total = pd.DataFrame(df.loc[:,feature].value_counts(dropna=False))
    ## concating percent and total dataframe

    total.columns = ["Total"]
    percent.columns = ['Percent']
    return pd.concat([total, percent], axis = 1)

missing_percentage(train)[missing_percentage(train)['Total'] != 0]    


# In[11]:


missing_percentage(test)[missing_percentage(test)['Total'] != 0]   


# In[12]:


percent_value_counts(train, 'Sex')


# In[13]:


pd.concat([percent_value_counts(train[train['Survived'] == 0], 'Sex'), percent_value_counts(train[train['Survived'] == 1], 'Sex')], axis=1, sort=False)


# In[14]:


train['Sex'] = train.Sex.apply(lambda x: 0 if x == 'male' else 1)
test['Sex'] = test.Sex.apply(lambda x: 0 if x == 'male' else 1)


# In[15]:


percent_value_counts(train, 'Cabin')


# In[16]:


percent_value_counts(train, 'Embarked')


# In[17]:


train[train['Embarked'].isna()]


# ### Completing / Deleting missing values in datasets

# In[18]:


train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)
drop_column = ['PassengerId', 'Cabin', 'Ticket']
train.drop(drop_column, axis=1, inplace=True)
test.drop(drop_column, axis=1, inplace=True)
print(train.isna().sum())
print(test.isna().sum())


# In[19]:


train.shape, test.shape


# In[20]:


print(train.columns[0])
print('-'*10)
for (i, j) in zip(train.columns[1:], test.columns):
    print(i)
    print(j)
    print('-'*10)


# In[21]:


train.info()


# In[22]:


percent_value_counts(train, 'Embarked')


# In[23]:


train.sample(6)


# In[24]:


def refill_Embarked(df):
    for i in range(len(df.Embarked)):
        if df.Embarked.loc[i] == 'S':
            df.Embarked.loc[i] = 1
        elif df.Embarked.loc[i] == 'C':
            df.Embarked.loc[i] = 2
        elif df.Embarked.loc[i] == 'Q':
            df.Embarked.loc[i] = 3


# In[25]:


refill_Embarked(train)


# In[26]:


refill_Embarked(test)


# In[27]:


train.drop('Name', axis=1, inplace=True); test.drop('Name', axis=1, inplace=True)


# In[28]:


print(train.shape)
train.info()


# In[29]:


print(test.shape)
test.info()


# In[30]:


percent_value_counts(train,'Embarked')


# In[31]:


plt.subplots(figsize = (12, 9))
sns.heatmap(train.corr(), 
            annot=True, 
            mask=np.zeros_like(train.corr(), dtype=np.bool),
            cmap = 'RdBu', 
            linewidths=.9, 
            linecolor='gray',
            fmt='.2g',
            center = 0,
            square=True
           
           );
plt.title("Correlations Among Features", y = 1.03,fontsize = 20, pad = 40);


# In[32]:


X = train.drop(['Survived'], axis = 1)
y = train["Survived"]


# In[33]:


X.shape, y.shape


# In[34]:


X.columns, test.columns


# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


# In[42]:


from sklearn.tree import DecisionTreeClassifier

des_tree_model = DecisionTreeClassifier(random_state=1)
des_tree_model.fit(X_train, y_train)


# In[44]:


from sklearn.metrics import mean_absolute_error
val_predict = model.predict(X_val)
print(mean_absolute_error(val_predict, y_val))


# In[45]:


print(mean_absolute_error(model.predict(X_test), y_test))


# In[47]:


from sklearn.linear_model import LogisticRegression
log_reg_model = LogisticRegression().fit(X_train, y_train)
print(mean_absolute_error(log_reg_model.predict(X_val), y_val))


# In[48]:


print(mean_absolute_error(model.predict(X_test), y_test))


# In[ ]:


from sklearn.linear_model import RidgeClassifier
ridge_class_model = RidgeClassifier().fit(X_train, y_train)
print(mean_absolute_error(ridge_class_model.predict(X_val), y_val))

