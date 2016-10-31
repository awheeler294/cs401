# color
# director_name
# num_critic_for_reviews
# duration
# director_facebook_likes
# actor_3_facebook_likes
# actor_2_name
# actor_1_facebook_likes
# gross
# genres
# actor_1_name
# movie_title
# num_voted_users
# cast_total_facebook_likes
# actor_3_name
# facenumber_in_poster
# plot_keywords
# movie_imdb_link
# num_user_for_reviews
# language
# country
# content_rating
# budget
# title_year
# actor_2_facebook_likes
# imdb_score
# aspect_ratio
# movie_facebook_likes

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pylab as plt
import statsmodels.formula.api as smf
from sklearn.cross_validation import train_test_split

### Load the dataset into dataframe, ensure read-in correctly
movieData = pd.read_csv("IMDB.csv")
movieData.head()
# movieData[['movie_title', 'gross']]

# movieData.dtypes

# movieData.columns

### make sure missing data read in as missing
# movieData[['DLHRWAGE']]
movieData = movieData.dropna()
# movieData[['DLHRWAGE']]

# remove rows with missing data (regression will fail to run with missing data)
# movieData = pd.read_csv("twins.txt", na_values=["."])
# movieData = movieData.dropna()
# movieData[['DLHRWAGE']]
# movieData.DLHRWAGE

### exploratory data analysis
### obtain summary statistics
### assess regression assumptions

movieData.describe()

# check normality of response variable (need to drop missing data to generate)
import matplotlib.pyplot as plt

# plt.hist(twinData.DLHRWAGE.dropna())
plt.hist(movieData.gross, 50)
plt.show()

### split data into training, test
movieTrain, movieTest = train_test_split(movieData, test_size=.3, random_state=123)
movieTrain.shape
movieTest.shape


# MUST SPECIFICALLY IDENTIFY CATEGORICAL VARIABLES
# (otherwise, they will be treated as continuous and estimated effect won't make sense)
# C(DMARRIED) is categorical
model = smf.ols(formula='DLHRWAGE ~ 1 + DEDUC1 + AGE + AGESQ + WHITEH + MALEH + EDUCH + \
                     WHITEL + MALEL + EDUCL + DEDUC2 + DTEN + C(DMARRIED) + C(DUNCOV)', data=movieTrain).fit()
model.summary()

pred = model.predict(movieTest)
print(np.array((pred - movieTest['DLHRWAGE']) ** 2).mean())

# loop while max p-value > .05:
#   run the regression
#   remove max p-value

# get highest p-value
print(model.pvalues.argmax())

# training error
print(model.mse_resid)



### conclusions?
