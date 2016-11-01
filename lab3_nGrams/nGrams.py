from __future__ import print_function

import sys
import string
from operator import add
from pyspark import SparkContext


def words_from_line(line):
    parts = line.lower().split('\t')
    parts[0] = ''.join([char for char in parts[0] if char not in string.punctuation and not char.isdigit()])
    return '\t'.join(parts)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: nGrams <input> <output>", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="nGrams")
    lines = sc.textFile(sys.argv[1], 1)
    counts = lines.map(words_from_line)

    counts.saveAsTextFile(sys.argv[2])
    sc.stop()
