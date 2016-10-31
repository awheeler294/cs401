from __future__ import print_function

import sys
import re
import string
from operator import add
from pyspark import SparkContext


def words_from_line(line):
    line = re.sub(r'[\d+:\d+]', '', line)
    return ''.join([char for char in line if char not in string.punctuation]) \
        .lower() \
        .split()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wordcount <input> <output>", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="WordCountAgain")
    lines = sc.textFile(sys.argv[1], 1)
    counts = lines.flatMap(words_from_line) \
        .map(lambda x: (x, 1)) \
        .reduceByKey(add) \
        .map(lambda p: (p[1], p[0])) \
        .sortByKey(0, 1) \
        .map(lambda p: (p[1], p[0]))

    counts.saveAsTextFile(sys.argv[2])
    sc.stop()
