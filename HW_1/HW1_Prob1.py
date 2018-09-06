
# coding: utf-8

# # USING THE F-STATISTIC

# In[1]:


import pandas as pd
import numpy as np
from sklearn import feature_selection
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_csv('/Users/rodney/Documents/Jupyter/HW_IE_691/HW_1/Electric_Power_Data.csv')
#print(df)


# In[3]:


columns = df[['x1', 'x2', 'x3', 'x4']]
print (columns)
df['y']


# In[4]:


model = feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=4)
results = model.fit(columns, df['y'])


# In[5]:


print (results.scores_)


# In[6]:


print (results.pvalues_)


# ### The desire is for lower p-values; the lower the p-value, the more significant the feature.
# ### Based on the p-values, the feature to drop is x4 (highest p-value).
# ### Based on the p-values, the best feature to keep is x2 (lowest p-value).
# ### Feature ranking based on p-value from most desired to keep - most desire to drop: x2, x1, x3, x4.

# # USING THE T-STATISTIC

# In[7]:


import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

diabetes = datasets.load_diabetes()
X = df[['x1', 'x2', 'x3', 'x4']]
y = df['y']

print (X)
print (y)


# In[8]:


lm = LinearRegression()
lm.fit(X,y)
params = np.append(lm.intercept_,lm.coef_)
predictions = lm.predict(X)

newX = np.append(np.ones((len(X),1)), X, axis=1)
MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/ sd_b

p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

sd_b = np.round(sd_b,3)
ts_b = np.round(ts_b,3)
p_values = np.round(p_values,3)
params = np.round(params,4)

myDF3 = pd.DataFrame()
myDF3["Coefficients"],myDF3["Std Errors"],myDF3["t values"],myDF3["p-values"] = [params,sd_b,ts_b,p_values]
print(myDF3)


# ### The desire is for lower p-values; the lower the p-value, the more significant the feature.
# ### Based on the p-values, the feature to drop is x4 (highest p-value).
# ### Based on the p-values, the best feature to keep is x2 (lowest p-value).
# ### Feature ranking based on p-value from most desired to keep - most desire to drop: x2, x1, x3, x4
