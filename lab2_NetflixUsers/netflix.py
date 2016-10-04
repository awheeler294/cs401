from __future__ import print_function

import sys
import re
import string
from operator import add
from pyspark import SparkContext


def data_from_line(line):
    user_data = line.split()

    return {'user_id': user_data[0], 'movie_id': user_data[1], 'rating': user_data[2]}


def map_user_data(target_user_rating, user_data):
    return user_data.filter(lambda x: (x['movie_id'] == target_user_rating['movie_id']
                                and x['rating'] == target_user_rating['rating']))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: netflix <input> <user> <output>", file=sys.stderr)
        exit(-1)
    targetUser = sys.argv[2]
    sc = SparkContext(appName="NetflixUsers")
    lines = sc.textFile(sys.argv[1], 1)

    userData = lines.flatMap(data_from_line)

    # get all movies targetUser rated
    targetUserRatings = userData.filter(lambda x: (x['user_id'] == targetUser))

    counts = targetUserRatings.flatMap(lambda target_user_rating: (map_user_data(target_user_rating, userData))) \
        .map(lambda x: (x, 1)) \
        .reduceByKey(add)

    counts = lines.flatMap(data_from_line) \
        .map(lambda x: (x, 1)) \
        .reduceByKey(add)

    counts.saveAsTextFile(sys.argv[3])
    sc.stop()
