from random import *
import sys

TEST_FILE_COUNT = int(sys.argv[1])
MAX_A = 1


def to_float_str(x):
    s = str(x / MAX_A)
    return s


ns = [2, 3, 1]
ls = [2, 2, 2]

grids = [12, 24, 48, 96, 144]

u = [3, 7, 2, 5, 1, 3, 10]

for test in range(1, TEST_FILE_COUNT + 1):
    testFile = open("tests/" + str(test) + ".in", "w")
    ansFile = open("tests/" + str(test) + ".out", "w")

    shuffle(ns)
    bs = []
    for i in range(3):
        # bs.append(grids[test - 1] // ns[i])
        bs.append(144 // ns[i])

    # .in
    testFile.write(" ".join(str(elem) for elem in ns) + "\n")
    testFile.write(" ".join(str(elem) for elem in bs) + "\n")
    testFile.write("report.txt\n")
    testFile.write("1e-4\n")
    testFile.write(" ".join(to_float_str(elem) for elem in ls) + "\n")
    testFile.write(" ".join(to_float_str(elem) for elem in u) + "\n")

    # .out
    # ansFile.write(" ".join(str(elem) for elem in [max(a)]) + "\n")

    testFile.close()
    ansFile.close()
