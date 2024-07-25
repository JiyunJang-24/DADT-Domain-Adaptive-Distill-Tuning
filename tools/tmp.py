import time


def heavy_work(name):
    result = 0
    for i in range(4000000):
        result += i
    print('%s done' % name)


if __name__ == '__main__':
    import multiprocessing

    start = time.time()
    pool = multiprocessing.Pool(processes=4)
    pool.map(heavy_work, range(4))
    pool.close()
    pool.join()

    end = time.time()

    print("수행시간: %f 초" % (end - start))