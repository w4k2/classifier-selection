from ARFF import ARFF
from DumbDelayPool import DumbDelayPool
from TestAndTrain import TestAndTrain
import matplotlib.pyplot as plt

streams = open("streams.txt", "r").read().split("\n")[:-1]

for filename in streams:
    # Przedstaw się, strumieniu
    print(filename)

    # Wczytaj się, strumieniu
    stream = ARFF("streams/" + filename)

    # Zainicjalizuj się, klasyfikatorze
    clf = DumbDelayPool(ensemble_size=10)

    # Przygotuj się, module uczący, poznając strumień i estymator
    learner = TestAndTrain(stream, clf)

    # Badaj
    learner.run()

    # Wyrysuj i zapisz
    plt.figure(figsize=(8, 4))
    plt.plot(learner.score_points, learner.scores, label='basic')
    plt.plot(learner.score_points, learner.des_scores, label='des')
    plt.title(filename)
    plt.legend()
    plt.ylim((0, 1))
    plt.savefig("figures/%s.png" % filename.split(".")[0])
    plt.savefig("foo.png")

    # Zapomnij, wyrzuć z pamięci
    stream.close()
