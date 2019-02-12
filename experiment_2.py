import csm
import matplotlib.pyplot as plt
import numpy as np
import helper as h
from tqdm import tqdm
import multiprocessing

# Select streams and methods
streams = h.streams()
clfs = h.clfs()

# Define worker
def worker(i, stream_n):
    """worker function"""
    stream = streams[stream_n]
    results = np.zeros((len(clfs), stream.n_chunks - 1))

    for j, clfn in enumerate(clfs):
        clf = clfs[clfn]

        print(
            "Starting clf %i/%i of stream %i/%i"
            % (j + 1, len(clfs), i + 1, len(streams))
        )

        learner = csm.TestAndTrain(stream, clf)
        learner.run()

        print(
            "Done clf %i/%i of stream %i/%i" % (j + 1, len(clfs), i + 1, len(streams))
        )

        results[j, :] = learner.scores

        stream.reset()

    np.save("results/experiment_2/%s" % stream, results)


jobs = []
for i, stream_n in enumerate(streams):
    p = multiprocessing.Process(target=worker, args=(i, stream_n))
    jobs.append(p)
    p.start()
