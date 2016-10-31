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
import statsmodels.formula.api as smf
from sklearn.cross_validation import train_test_split

 
### Load the dataset into dataframe, ensure read-in correctly
twinData = pd.read_csv("twins.txt")
twinData.head()
twinData[['DLHRWAGE','EDUCH']]

twinData.dtypes

twinData.columns

### make sure missing data read in as missing
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
#plt.hist(twinData.DLHRWAGE.dropna())
plt.hist(twinData.DLHRWAGE,50)
plt.show()



### split data into training, test
twinTrain, twinTest = train_test_split(twinData, test_size=.3, random_state=123)
twinTrain.shape
twinTest.shape



### basic linear regression (without variable selection)

# adding '1' as a predictor forces the inclusion of an intercept
model = smf.ols(formula='DLHRWAGE ~ 1 + DEDUC1 + AGE + AGESQ + WHITEH + MALEH + EDUCH + \
                     WHITEL + MALEL + EDUCL + DEDUC2 + DTEN + DMARRIED + DUNCOV', data=twinTrain).fit()
model.summary()

# MUST SPECIFICALLY IDENTIFY CATEGORICAL VARIABLES
# (otherwise, they will be treated as continuous and estimated effect won't make sense)
model = smf.ols(formula='DLHRWAGE ~ 1 + DEDUC1 + AGE + AGESQ + WHITEH + MALEH + EDUCH + \
                     WHITEL + MALEL + EDUCL + DEDUC2 + DTEN + C(DMARRIED) + C(DUNCOV)', data=twinTrain).fit()
model.summary()



### conclusions?

