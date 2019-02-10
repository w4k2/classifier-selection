from Method import Method
from MethodAlternate import MethodAlternate
from Dumb import Dumb
from TestAndTrain import TestAndTrain
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import helper as h
from tqdm import tqdm

np.set_printoptions(suppress=True)

streams = h.streams()

clfs = {
    "MIN": MethodAlternate(alpha=0.2, ensemble_size=5, decision="min"),
    "BAS": MethodAlternate(alpha=0.2, ensemble_size=5, decision="basic"),
    "E": Method(),
    "DUMB": Dumb(),
}


for i, stream_n in tqdm(enumerate(streams), total=len(streams), ascii=True):
    plt.figure(figsize=(8, 4))

    stream = streams[stream_n]
    results = np.zeros((6, stream.n_chunks - 1))
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
        "results/%s.csv" % stream, results.T, fmt="%.0f, %.3f, %.3f, %.3f, %.3f, %.3f"
    )

    plt.legend()
    plt.savefig("foo.png")
    plt.savefig("figures/%s.png" % stream)

    plt.close()
