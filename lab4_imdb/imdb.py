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
import matplotlib.pyplot as plt
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

# plt.hist(twinData.DLHRWAGE.dropna())
plt.hist(movieData.gross, 50)
# plt.show()

recode1 = {"R": 1, "X": 1, "NC-17": 1, "G": 0, "PG": 0, "PG-13": 0}
recode2 = {"R": 0, "X": 0, "NC-17": 0, "G": 1, "PG": 1, "PG-13": 1}

# transform levels of categorical variablies into 0/1 dummy variables
movieData['rating_restricted'] = movieData.content_rating.map(recode1)
movieData['rating_not_restricted'] = movieData.content_rating.map(recode2)

### split data into training, test
movieTrain, movieTest = train_test_split(movieData, test_size=.3, random_state=123)
movieTrain.shape
movieTest.shape

# MUST SPECIFICALLY IDENTIFY CATEGORICAL VARIABLES
# (otherwise, they will be treated as continuous and estimated effect won't make sense)
predictors = ['C(color)', 'duration', 'C(content_rating)', 'budget', 'title_year', 'num_critic_for_reviews',
              'director_facebook_likes', 'actor_1_facebook_likes', ' actor_2_facebook_likes', 'actor_3_facebook_likes',
              'cast_total_facebook_likes', 'movie_facebook_likes', 'facenumber_in_poster']
pValue = 1
while pValue > .005:
    model = smf.ols(formula='gross ~ 1 + ' + ' + '.join(predictors),
                    data=movieTrain).fit()

    argMax = model.pvalues.argmax()
    pValue = model.pvalues
    # get highest p-value
    print("\nHighest P-value:")
    print(argMax)
    print(pValue)


print(model.summary())

pred = model.predict(movieTest)
print("\nPrediction Error:")
print(np.array((pred - movieTest['gross']) ** 2).mean())

# loop while max p-value > .05:
#   run the regression
#   remove max p-value

# training error
print("\nTraining Error:")
print(model.mse_resid)

### conclusions?
