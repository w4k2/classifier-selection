import csm
import matplotlib.pyplot as plt
import numpy as np
import helper as h
from tqdm import tqdm

streams = h.streams()

clfs = {
    "MDE": csm.MDE(decision="min", ensemble_size=3, alpha=0.05),
    "MDEb": csm.MDE(decision="basic", ensemble_size=3, alpha=0.05),
    "KNORAE": csm.DESlibStream(desMethod="KNORAE"),
    "KNORAU": csm.DESlibStream(desMethod="KNORAU"),
    "Rank": csm.DESlibStream(desMethod="Rank"),
    "LCA": csm.DESlibStream(desMethod="LCA"),
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
