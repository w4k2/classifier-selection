from Method import Method
from TestAndTrain import TestAndTrain
from StreamGenerator import StreamGenerator
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


d = [0.1, 0.9]
chunk_size = 500
n_chunks = 200
n_features = 8
n_drifts = 4

class_seps = [0.0, 0.3, 0.6, 1.0]
drift_types = ["incremental", "sudden"]

rs = 0

streams = {}
for drift_type in drift_types:
    rs += 1
    for class_sep in class_seps:
        stream = StreamGenerator(
            drift_type=drift_type,
            random_state=rs,
            distribution=d,
            n_chunks=n_chunks,
            n_features=n_features,
            n_drifts=n_drifts,
            class_sep=class_sep,
        )
        streams.update({str(stream): stream})


clfs = {
    "MET1": Method(ensemble_size=1),
    "MET5": Method(ensemble_size=5),
    "MLP": MLPClassifier(),
}


for i, stream_n in enumerate(streams):
    plt.figure(figsize=(8, 4))

    stream = streams[stream_n]
    plt.ylim((0, 1))
    plt.title(stream)

    for clfn in clfs:
        clf = clfs[clfn]
        learner = TestAndTrain(stream, clf)
        learner.run()

        # Wyrysuj i zapisz
        plt.plot(learner.score_points, learner.scores, label=clfn)

        stream.reset()
    plt.legend()
    plt.savefig("foo.png")
    plt.savefig("figures/%s.png" % stream)
    print("![](figures/%s.png)\n" % stream)
    plt.close()
