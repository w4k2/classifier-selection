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


"""
# Process
for i, stream_n in tqdm(enumerate(streams), total=len(streams), ascii=True):
    print(stream_n)
    plt.figure(figsize=(8, 4))

    stream = streams[stream_n]
    results = np.zeros((7, stream.n_chunks - 1))
    plt.ylim((0.4, 1))
    plt.title(stream)

    for j, clfn in tqdm(enumerate(clfs), ascii=True, total=len(clfs)):
        clf = clfs[clfn]
        learner = csm.TestAndTrain(stream, clf)
        learner.run()

        # Wyrysuj i zapisz
        plt.plot(learner.score_points, learner.scores, label=clfn)
        results[0, :] = learner.score_points
        results[j + 1, :] = learner.scores

        stream.reset()

    np.savetxt(
        "results/%s.csv" % stream,
        results.T,
        fmt="%.0f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f",
    )

    plt.legend()
    plt.savefig("foo.png")
    # exit()
    plt.savefig("figures/run/%s.png" % stream)

    plt.close()
    exit()
"""
