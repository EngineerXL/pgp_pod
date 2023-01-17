from random import *

TEST_FILE_COUNT = 8
MIN_N = 10 ** 8
MAX_N = 10 ** 8

MAX_A = 10000

def to_float_str(x):
    s = str(x / MAX_A)
    return s

ns = [10, 100, 1000, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8]

for test in range(1, TEST_FILE_COUNT + 1):
    testFile = open("tests/" + str(test) + ".in", "w")
    ansFile = open("tests/" + str(test) + ".out", "w")

    n = ns[test - 1]
    # n = randint(MIN_N, MAX_N)
    a = [randint(-MAX_A, MAX_A) for _ in range(n)]

    # a = [randint(-MAX_A, MAX_A) for _ in range(n)]
    # for _ in range(n - 1):
    #     a.append(a[-1] + 5000)

    # .in
    testFile.write(" ".join(str(elem) for elem in [n]) + "\n")
    testFile.write(" ".join(to_float_str(elem) for elem in a) + "\n")

    # .out
    # ansFile.write(" ".join(str(elem) for elem in [max(a)]) + "\n")

    testFile.close()
    ansFile.close()
