from __future__ import print_function

import sys
import re
import string
from operator import add
from pyspark import SparkContext

USER_ID = 0
MOVIE_ID = 1
RATING = 2


def index_by_movie_id(line):
    user_data = line.split()

    return user_data[MOVIE_ID], [user_data[RATING], user_data[USER_ID]]


def map_user_data(target_user_rating, user_data):
    return user_data.filter(lambda x: (lambda y=x.split(): (y[MOVIE_ID] == target_user_rating[MOVIE_ID]
                                                            and y[RATING] == target_user_rating[RATING])))


def filter_by_same_rating(line):
    # movie_id = line[0]
    # rating = line[1][0]
    line = line.split()
    for user_rating in ratings.value:
        user_rating = user_rating.split()
        if user_rating[MOVIE_ID] == line[MOVIE_ID] and user_rating[RATING] == line[RATING]:
            return True

    return False


def filter_by_target_user(line):
    global targetUser
    line_data = line.split()
    return line_data[USER_ID] == targetUser


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: netflix <input> <user> <output>", file=sys.stderr)
        exit(-1)
    global targetUser

    inputFile = sys.argv[1]
    print(inputFile)

    targetUser = sys.argv[2]
    print(targetUser)

    outputFile = sys.argv[3]
    print(outputFile)

    sc = SparkContext(appName="NetflixUsers")
    lines = sc.textFile(inputFile, 1)

    # userData = lines.flatMap(data_from_line)

    # get all movies targetUser rated
    # targetUserRatings = lines.filter(filter_by_target_user)

    # counts = targetUserRatings.flatMap(lambda target_user_rating: (map_user_data(target_user_rating, lines))) \
    #     .map(lambda x: (x, 1)) \
    #     .reduceByKey(add)
    userRatings = lines.filter(filter_by_target_user)

    # userRatings.saveAsTextFile(outputFile)

    # movieIdIndex = lines.map(index_by_movie_id)

    ratings = sc.broadcast(userRatings.collect())

    sameRating = lines.filter(filter_by_same_rating) \
        .flatMap(lambda x: (x.split()[0]))\
        .map(lambda x: (x, 1)) \
        .reduceByKey(add) \
        .map(lambda p: (p[1], p[0])) \
        .sortByKey(0, 1) \
        .map(lambda p: (p[1], p[0]))

    # userRatings.map(lambda x: (x[MOVIE_ID], x[RATING])).saveAsTextFile(outputFile)
    sameRating.saveAsTextFile(outputFile)
    # movie_id_index.saveAsTextFile(outputFile)

    # counts.saveAsTextFile(sys.argv[3])
    sc.stop()
