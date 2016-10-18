from __future__ import print_function

import sys
import re
import string
from operator import add
from pyspark import SparkContext

USER_ID = 0
MOVIE_ID = 1
RATING = 2


def data_from_line(line):
    user_data = line.split()

    return user_data[MOVIE_ID], [user_data[RATING], user_data[USER_ID]]


def map_user_data(target_user_rating, user_data):
    return user_data.filter(lambda x: (lambda y=x.split(): (y[MOVIE_ID] == target_user_rating[MOVIE_ID]
                                                            and y[RATING] == target_user_rating[RATING])))


def filter_by_same_rating(line, rating):
    return


def filter_by_target_user(line):
    global targetUser
    line_data = line.split()
    return line_data[USER_ID] == targetUser


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: netflix <input> <user> <output>", file=sys.stderr)
        exit(-1)
    global targetUser
    targetUser = sys.argv[2]
    # print(targetUser)
    sc = SparkContext(appName="NetflixUsers")
    lines = sc.textFile(sys.argv[1], 1)

    # userData = lines.flatMap(data_from_line)

    # get all movies targetUser rated
    # targetUserRatings = lines.filter(filter_by_target_user)

    # counts = targetUserRatings.flatMap(lambda target_user_rating: (map_user_data(target_user_rating, lines))) \
    #     .map(lambda x: (x, 1)) \
    #     .reduceByKey(add)

    counts = lines.flatMap(data_from_line) \
        .reduceByKey(1)

    counts.saveAsTextFile(sys.argv[3])

    # counts.saveAsTextFile(sys.argv[3])
    sc.stop()
