from DumbDelayPool import DumbDelayPool
from TestAndTrain import TestAndTrain
from StreamGenerator import StreamGenerator
import matplotlib.pyplot as plt

streams = {
    "Sudden": StreamGenerator(n_features=4, drift="sudden", distribution=[0.3, 0.7]),
    "Incremental": StreamGenerator(
        n_features=4, drift="incremental", distribution=[0.3, 0.7]
    ),
}

fig, ax = plt.subplots(2, 1, figsize=(8, 8))

for i, stream_n in enumerate(streams):
    stream = streams[stream_n]
    clf = DumbDelayPool(ensemble_size=5)
    learner = TestAndTrain(stream, clf)
    learner.run()

    # Wyrysuj i zapisz
    ax[i].plot(learner.score_points, learner.scores)
    ax[i].plot(learner.score_points, (stream.b % 2 == 0)[:-1])
    ax[i].set_ylim((0, 1))
    ax[i].set_title(stream_n)

plt.savefig("foo.png")
