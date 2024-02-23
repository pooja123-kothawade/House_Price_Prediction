#!/usr/bin/env python
# coding: utf-8

# ## Dragon Real Estate- Price Predictor

# In[1]:

import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['4. CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


# import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(20,15))


# ## TRAIN-TEST SPLITTING

# In[9]:


# import numpy as np
# def split_train_test(data, test_ratio):
#     np.random.seed(50)
#     shuffled=np.random.permutation(len(data))
#     print shuff
#     test_set_size=int(len(data)*test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices=shuffled[test_set_size:]
#     return data.iloc[train_indices],data.iloc[test_indices]


# In[10]:


# train_set, test_set = split_train_test(housing, 0.2)


# In[11]:


# print(f"Rows in train set: {len(train_set)}\nRows in test set:{len(test_set)}\n")


# In[12]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['4. CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[14]:


strat_test_set['4. CHAS'].value_counts()


# In[15]:


strat_train_set['4. CHAS'].value_counts()


# In[16]:


# 95/7


# In[17]:


# 376/28


# In[18]:


housing = strat_train_set.copy()


# ## Looking for Correlations

# In[19]:


corr_matrix = housing.corr()
corr_matrix['14. MEDV'].sort_values(ascending=False)


# In[20]:


#from pandas.plotting import scatter_matrix
#attributes = ["14. MEDV", "6. RM", "2. ZN", "13. LSTAT"]
#scatter_matrix(housing[attributes], figsize = (12,8))


# In[21]:


housing.plot(kind="scatter", x="6. RM", y="14. MEDV", alpha=0.8)


# ## Trying out Attribute combinations

# In[22]:


housing["TAXRM"] = housing['10. TAX']/housing['6. RM']


# In[23]:


housing.head()


# In[24]:


corr_matrix = housing.corr()
corr_matrix['14. MEDV'].sort_values(ascending=False)


# In[25]:


housing.plot(kind="scatter", x="TAXRM", y="14. MEDV", alpha=0.8)


# In[26]:


housing = strat_train_set.drop("14. MEDV", axis=1)
housing_labels = strat_train_set["14. MEDV"].copy()


# ## Missing Attributes##

# In[27]:


# To take care of missing attributes, you have three options:
#     1. Get rid of the missing data points
#     2. Get rid of the whole attribute
#     3. Set the value to some value(0, mean or median)


# In[28]:


a = housing.dropna(subset=["6. RM"]) #Option 1
a.shape
# Note that the original housing dataframe will remain unchanged


# In[29]:


housing.drop("6. RM", axis=1).shape # Option 2
# Note that there is no RM column and also note that the original housing dataframe will remain unchanged


# In[30]:


median = housing["6. RM"].median() # Compute median for Option 3
median


# In[31]:


housing["6. RM"].fillna(median) # Option 3
# Note that the original housing dataframe will remain unchanged


# In[32]:


housing.shape


# In[33]:


housing.describe() # before we started filling missing attributes


# In[34]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)


# In[35]:


#SimpleImputer(add_indicator=False, copy=True, fill_value=None,missing_values=nan, strategy='median', verbose=0)


# In[36]:


imputer.statistics_.shape


# In[37]:


X = imputer.transform(housing)


# In[38]:


housing_tr = pd.DataFrame(X, columns=housing.columns)


# In[39]:


housing_tr.describe()


# ## Scikit-learn Design

# ## Feature Scaling

# ## Creating a Pipeline

# In[40]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")), #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])


# In[41]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[42]:


housing_num_tr


# ## Selecting a desired model for Dragon Real Estates

# In[43]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[44]:


#RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
 #                     max_features='auto', max_leaf_nodes=None,
  #                    min_impurity_decrease=0.0, min_impurity_split=None,
   #                   min_samples_leaf=1, min_samples_split=2,
    #                  min_weight_fraction_leaf=0.0, n_estimators=10,
     #                 n_jobs=None, oob_score=False, random_state=None,
      #                verbose=0, warm_start=False)


# In[45]:


some_data = housing.iloc[:5]


# In[46]:


some_labels = housing_labels.iloc[:5]


# In[47]:


prepared_data = my_pipeline.transform(some_data)


# In[48]:


model.predict(prepared_data)


# In[49]:


list(some_labels)


# ##Evaluating the model

# In[50]:


import numpy as np


# In[51]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[52]:


rmse


# ##using better evaluation technique - cross validation

# In[53]:


# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[54]:


rmse_scores


# In[55]:


#array([3.04485171, 2.48131898, 4.63312016, 2.8778676 , 3.41281409,
 #      3.03586684, 4.85712775, 3.52571837, 2.89743852, 4.18037857])


# In[56]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# In[57]:


print_scores(rmse_scores)


# Quiz: Convert this notebook into a python file and run the pipeline using Visual Studio Code
# 

# ## Saving the model

# In[58]:


from joblib import dump, load
dump(model, 'Dragon.joblib') 


# ## testing the model on test data

# In[59]:


X_test = strat_test_set.drop("14. MEDV", axis=1)
Y_test = strat_test_set["14. MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
#print(final_predictions, list(Y_test))


# In[60]:


final_rmse


# In[61]:


prepared_data[0]


# ## Using the model

# In[62]:


from joblib import dump, load
import numpy as np
model = load('Dragon.joblib') 
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])
model.predict(features)


# In[ ]:




