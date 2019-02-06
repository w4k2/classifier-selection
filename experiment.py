from Method import Method
from TestAndTrain import TestAndTrain
from StreamGenerator import StreamGenerator
import matplotlib.pyplot as plt

d = [0.3, 0.7]
streams = {
    "Sudden": StreamGenerator(
        n_features=4, drift_type="sudden", distribution=d, n_chunks=50, random_state=0
    ),
    "Incremental": StreamGenerator(
        n_features=4,
        drift_type="incremental",
        distribution=d,
        n_chunks=50,
        random_state=0,
    ),
}

fig, ax = plt.subplots(2, 1, figsize=(8, 8))

for i, stream_n in enumerate(streams):
    stream = streams[stream_n]
    clf = Method(ensemble_size=5)
    learner = TestAndTrain(stream, clf)
    learner.run()

    # Wyrysuj i zapisz
    ax[i].plot(learner.score_points, learner.scores)
    ax[i].set_ylim((0, 1))
    ax[i].set_title(stream_n)
    # ax[i].plot(learner.score_points, (stream.concept_dominances % 2 == 0)[:-1])
    # ax[i].plot(learner.score_points, stream.usage_curve[:-1] / stream.chunk_size)

    print(stream.concept_usages)

    stream.reset()

plt.savefig("foo.png")
