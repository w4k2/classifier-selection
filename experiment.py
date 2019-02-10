from Method import Method
from MethodAlternate import MethodAlternate
from DESlibStream import DESlibStream
from TestAndTrain import TestAndTrain
from Dumb import Dumb
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import helper as h
from tqdm import tqdm

from deslib.des import KNORAE, KNORAU
from deslib.dcs import Rank, LCA

streams = h.streams()

clfs = {
    "MET_ALT": MethodAlternate(),
    "KNORAE": DESlibStream(desMethod="KNORAE"),
    "KNORAU": DESlibStream(desMethod="KNORAU"),
    "Rank": DESlibStream(desMethod="Rank"),
    "LCA": DESlibStream(desMethod="LCA"),
    "Dumb": Dumb(),
}


for i, stream_n in tqdm(enumerate(streams), total=len(streams), ascii=True):
    plt.figure(figsize=(8, 4))

    stream = streams[stream_n]
    results = np.zeros((7, stream.n_chunks - 1))
    plt.ylim((0.4, 1))
    plt.title(stream)

    for j, clfn in tqdm(enumerate(clfs), ascii=True, total=len(clfs)):
        clf = clfs[clfn]
        learner = TestAndTrain(stream, clf)
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
    plt.savefig("figures-des/%s.png" % stream)

    plt.close()
    exit()
