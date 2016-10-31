# -*- coding: utf-8 -*-


# DLHRWAGE.....the difference (twin 1 minus twin 2) in the logarithm of 
#              hourly wage, given in dollars.                                              
# DEDUC1.......the difference (twin 1 minus twin 2) in self-reported 
#  	       education, given in years.                                        
# AGE..........Age in years of twin 1.                                                 
# AGESQ........AGE squared. 
# HRWAGEH......Hourly wage of twin 2. 
# WHITEH.......1 if twin 2 is white, 0 otherwise.
# MALEH........1 if twin 2 is male, 0 otherwise.
# EDUCH........Self-reported education (in years) of twin 2.
# HRWAGEL......Hourly wage of twin 1.
# WHITEL.......1 if twin 1 is white, 0 otherwise.
# MALEL........1 if twin 1 is male, 0 otherwise.
# EDUCL........Self-reported education (in years) of twin 1.
# DEDUC2.......the difference (twin 1 minus twin 2) in cross-reported 
#              education. Twin 1's cross-reported education, for example,
#              is the number of years of schooling completed by twin 1 as
#              reported by twin 2.                                     
# DTEN.........the difference (twin 1 minus twin 2) in tenure, or number of 
#	       years at current job.
# DMARRIED.....the difference (twin 1 minus twin 2) in marital status, where
#              1 signifies "married" and 0 signifies "unmarried". 
# DUNCOV........the difference (twin 1 minus twin 2) in union coverage, where
#              1 signifies "covered" and 0 "uncovered". 


import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV

### Load the dataset into dataframe, ensure read-in correctly
twinData = pd.read_csv("twins.txt")
twinData.head()

twinData.dtypes

twinData.columns

### make sure missing data read in as missing
twinData[['DLHRWAGE', 'EDUCH']]
twinData[['DLHRWAGE']]
twinData = twinData.dropna()
twinData[['DLHRWAGE']]

# remove rows with missing data (regression will fail to run with missing data)
twinData = pd.read_csv("twins.txt", na_values=["."])
twinData = twinData.dropna()
twinData[['DLHRWAGE']]
twinData.DLHRWAGE

### exploratory data analysis
### obtain summary statistics
### assess regression assumptions

twinData.describe()

# check normality of response variable (need to drop missing data to generate)
import matplotlib.pyplot as plt

# plt.hist(twinData.DLHRWAGE.dropna())
plt.hist(twinData.DLHRWAGE, 50)
plt.show()

### basic linear regression (without variable selection)

import statsmodels.api as sm

# if I needed to convert one of my variables to factors, could do so
twinData.MALEL
twinData['MALEL'] = pd.Categorical(twinData.MALEL).codes  # convert text categories to numeric codes
X = twinData.drop('DLHRWAGE', axis=1)
X.columns
y = twinData[['DLHRWAGE']]

# include intercept in model
X1 = sm.add_constant(X)

model = sm.OLS(y, X1).fit()
model.summary()

recode1 = {-1: 0, 0: 0, 1: 1}
recode2 = {-1: 0, 0: 1, 1: 0}
recode3 = {-1: 1, 0: 0, 1: 0}

# transform levels of categorical variablies into 0/1 dummy variables
twinData['DMARRIED1'] = twinData.DMARRIED.map(recode1)
twinData['DMARRIED0'] = twinData.DMARRIED.map(recode2)
twinData.head(10)

twinData['DUNCOV1'] = twinData.DUNCOV.map(recode1)
twinData['DUNCOV0'] = twinData.DUNCOV.map(recode2)
twinData.head(10)

# select predictor variables and target variable as separate data sets
predvar = twinData[
    ['DEDUC1', 'AGE', 'AGESQ', 'WHITEH', 'MALEH', 'EDUCH', 'WHITEL', 'MALEL', 'EDUCL', 'DEDUC2', 'DTEN', 'DMARRIED0',
     'DMARRIED1', 'DUNCOV0', 'DUNCOV1']]

target = twinData.DLHRWAGE

# normalize
target = twinData.DLHRWAGE.apply(math.log)

# standardize predictors to have mean=0 and sd=1 (required for LASSO)
predictors = predvar.copy()
from sklearn import preprocessing

predictors['DEDUC1'] = preprocessing.scale(predictors['DEDUC1'].astype('float64'))
predictors['AGE'] = preprocessing.scale(predictors['AGE'].astype('float64'))
predictors['AGESQ'] = preprocessing.scale(predictors['AGESQ'].astype('float64'))
predictors['WHITEH'] = preprocessing.scale(predictors['WHITEH'].astype('float64'))
predictors['MALEH'] = preprocessing.scale(predictors['MALEH'].astype('float64'))
predictors['EDUCH'] = preprocessing.scale(predictors['EDUCH'].astype('float64'))
predictors['WHITEL'] = preprocessing.scale(predictors['WHITEL'].astype('float64'))
predictors['MALEL'] = preprocessing.scale(predictors['MALEL'].astype('float64'))
predictors['EDUCL'] = preprocessing.scale(predictors['EDUCL'].astype('float64'))
predictors['DEDUC2'] = preprocessing.scale(predictors['DEDUC2'].astype('float64'))
predictors['DTEN'] = preprocessing.scale(predictors['DTEN'].astype('float64'))
predictors['DMARRIED0'] = preprocessing.scale(predictors['DMARRIED0'].astype('float64'))
predictors['DMARRIED1'] = preprocessing.scale(predictors['DMARRIED1'].astype('float64'))
predictors['DUNCOV0'] = preprocessing.scale(predictors['DUNCOV0'].astype('float64'))
predictors['DUNCOV1'] = preprocessing.scale(predictors['DUNCOV1'].astype('float64'))

# split data into train and test sets
pred_train, pred_test, resp_train, resp_test = train_test_split(predictors, target,
                                                                test_size=.3, random_state=123)

# specify the lasso regression model
# precompute=True helpful for large data sets
model = LassoLarsCV(cv=10, precompute=True).fit(pred_train, resp_train)

# print variable names and regression coefficients
# note: we standardized variables so we can look at size of coefficients to assess which variables have the most predictive power
dict(zip(predictors.columns, model.coef_))

# plot coefficient progression
# note: Python refers to LASSO penalty (i.e, tuning) parameter as alpha
# note: apply -log(10) transformation to alpha values simply to make them easier to read
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()  # set up axes
plt.plot(m_log_alphas, model.coef_path_.T)  # alpha on x axis, change in regression coefficients on y axis
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')  # plot black dashed line at alpha value ultimately selected
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')
plt.savefig('Fig01')

# plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
plt.savefig('Fig02')

# MSE from training and test data
from sklearn.metrics import mean_squared_error

train_error = mean_squared_error(resp_train, model.predict(pred_train))
test_error = mean_squared_error(resp_test, model.predict(pred_test))
print('training data MSE')
print(train_error)
print('test data MSE')
print(test_error)

# R-square from training and test data
rsquared_train = model.score(pred_train, resp_train)
rsquared_test = model.score(pred_test, resp_test)
print('training data R-square')
print(rsquared_train)
print('test data R-square')
print(rsquared_test)
