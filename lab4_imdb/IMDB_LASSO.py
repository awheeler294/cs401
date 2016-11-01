import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV

### Load the dataset into dataframe, ensure read-in correctly
movieData = pd.read_csv("IMDB.csv")
movieData.head()

### make sure missing data read in as missing
movieData = movieData.dropna()

### exploratory data analysis
### obtain summary statistics
### assess regression assumptions

movieData.describe()

# check normality of response variable (need to drop missing data to generate)
import matplotlib.pyplot as plt

# plt.hist(twinData.DLHRWAGE.dropna())
plt.hist(movieData.gross, 50)
plt.show()

### basic linear regression (without variable selection)

import statsmodels.api as sm
# if I needed to convert one of my variables to factors, could do so
movieData['color'] = pd.Categorical(movieData.color).codes  # convert text categories to numeric codes
movieData['content_rating'] = pd.Categorical(movieData.content_rating).codes  # convert text categories to numeric codes
# X = movieData.drop('gross', axis=1)
# y = movieData[['gross']]
#
# # include intercept in model
# X1 = sm.add_constant(X)
#
# model = sm.OLS(y, X1).fit()
# model.summary()

# recode1 = {'Color': 1, 'Black and White': 0}
# recode2 = {'R': 1, 'NC-17': 1, 'X': 1, 'G': 0, 'PG': 0, 'PG-13': 0}
#
# # transform levels of categorical variablies into 0/1 dummy variables
# movieData['color1'] = movieData.color.map(recode1)
# movieData.head(10)
#
# movieData['rating_restricted'] = movieData.content_rating.map(recode2)
# movieData.head(10)

# select predictor variables and target variable as separate data sets
predvar = movieData[
    ['color', 'duration', 'content_rating', 'budget', 'title_year', 'num_critic_for_reviews',
     'director_facebook_likes', 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes',
     'cast_total_facebook_likes', 'movie_facebook_likes', 'facenumber_in_poster']]

target = movieData.gross

# normalize
# target = movieData.DLHRWAGE.apply(math.log)

# standardize predictors to have mean=0 and sd=1 (required for LASSO)
predictors = predvar.copy()
from sklearn import preprocessing

predictors['color'] = preprocessing.scale(predictors['color'].astype('float64'))
predictors['duration'] = preprocessing.scale(predictors['duration'].astype('float64'))
predictors['content_rating'] = preprocessing.scale(predictors['content_rating'].astype('float64'))
predictors['budget'] = preprocessing.scale(predictors['budget'].astype('float64'))
predictors['title_year'] = preprocessing.scale(predictors['title_year'].astype('float64'))
predictors['num_critic_for_reviews'] = preprocessing.scale(predictors['num_critic_for_reviews'].astype('float64'))
predictors['director_facebook_likes'] = preprocessing.scale(predictors['director_facebook_likes'].astype('float64'))
predictors['actor_1_facebook_likes'] = preprocessing.scale(predictors['actor_1_facebook_likes'].astype('float64'))
predictors['actor_2_facebook_likes'] = preprocessing.scale(predictors['actor_2_facebook_likes'].astype('float64'))
predictors['actor_3_facebook_likes'] = preprocessing.scale(predictors['actor_3_facebook_likes'].astype('float64'))
predictors['cast_total_facebook_likes'] = preprocessing.scale(predictors['cast_total_facebook_likes'].astype('float64'))
predictors['movie_facebook_likes'] = preprocessing.scale(predictors['movie_facebook_likes'].astype('float64'))
predictors['facenumber_in_poster'] = preprocessing.scale(predictors['facenumber_in_poster'].astype('float64'))

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
