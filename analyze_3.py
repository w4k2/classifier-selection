"""
Hyperparameters overview
"""
import numpy as np
import matplotlib.pyplot as plt
import helper as h
import csm
import seaborn as sb
from scipy.ndimage.filters import gaussian_filter1d

np.set_printoptions(precision=3)
p = 0.05

cs = ["black", "red", "red", "blue", "blue"]
# cs = ["red", "blue", "blue", "black", "black"]
ls = ["-", "-", "--", "-", "--"]

# Select streams and methods
streams = h.streams()
clfs = h.clfs()

# Stream Variables
drift_types = ["incremental", "sudden"]
distributions = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]
random_states = [1337, 666, 42]
label_noises = [0.0, 0.1, 0.2, 0.3]


ldistributions = [[0.1, 0.9], [0.2, 0.8]]

# Prepare storage for results
chunk_size = next(iter(streams.values())).chunk_size
n_chunks = next(iter(streams.values())).n_chunks
# score_points = list(range(chunk_size, chunk_size * n_chunks, chunk_size))


def gather_and_present(title, filename, streams, what):
    results_hypercube = np.zeros((len(streams), len(clfs), n_chunks - 1))
    for i, stream_n in enumerate(streams):
        results = np.load("results/experiment_3/%s.npy" % stream_n)
        results_hypercube[i] = results

    overall = np.mean(results_hypercube, axis=0)

    plt.figure(figsize=(8, 5))

    for j, clfn in enumerate(clfs):
        clf = clfs[clfn]
        val = gaussian_filter1d(overall[j], sigma=1, mode="nearest")
        plt.plot(val, label=clfn, ls=ls[j], c=cs[j])

    plt.ylim((0.5, 1))
    plt.xlim(0, 200)
    # plt.xlabel("chunks", fontsize=12)
    # plt.ylabel("Balanced accuracy", fontsize=12)
    plt.xticks(fontfamily="serif", fontsize=13)
    plt.yticks(fontfamily="serif", fontsize=13)
    plt.ylabel("BAC", fontfamily="serif", fontsize=14)
    plt.xlabel("chunks", fontfamily="serif", fontsize=14)

    # plt.yticks(
    #     [0.5, 0.6, 0.7, 0.8, 0.9, 1],
    #     ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
    #     fontsize=12,
    # )
    #
    # plt.xticks(
    #     [0, 25, 50, 75, 100, 125, 150, 175, 200],
    #     ["0", "25", "50", "75", "100", "125", "150", "175", "200"],
    #     fontsize=12,
    # )

    plt.legend(loc=9, ncol=6, columnspacing=1, frameon=False)
    plt.title(title, fontsize=18, fontfamily="serif")
    plt.grid(ls="--", color=(0.85, 0.85, 0.85))
    plt.tight_layout()
    sb.despine(top=True, right=True, left=False, bottom=False)
    # plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")

    a = np.swapaxes(results_hypercube, 1, 0)
    res = np.reshape(a, (5, -1))

    return h.tabrow(what, res)


# Compare drift types
print("Drift types")
text_file = open("rows/drift_types.tex", "w")
for drift_type in drift_types:
    streams = {}
    for distribution in ldistributions:
        for random_state in random_states:
            for flip_y in label_noises:
                stream = csm.StreamGenerator(
                    drift_type=drift_type,
                    distribution=distribution,
                    random_state=random_state,
                    flip_y=flip_y,
                )
                streams.update({str(stream): stream})

    title = drift_type + " drift"
    filename = "figures/experiment_3_%s" % drift_type
    what = drift_type
    tabrow = gather_and_present(title, filename, streams, what)
    print(tabrow)
    text_file.write(tabrow + "\n")
text_file.close()


# Compare distributions
print("Distributions")
text_file = open("rows/distributions.tex", "w")
for distribution in distributions:
    streams = {}
    for drift_type in drift_types:
        for random_state in random_states:
            for flip_y in label_noises:
                stream = csm.StreamGenerator(
                    drift_type=drift_type,
                    distribution=distribution,
                    random_state=random_state,
                    flip_y=flip_y,
                )
                streams.update({str(stream): stream})

    title = "%i%% of minority class" % int(distribution[0] * 100)
    what = "%.0f\\%%" % (distribution[0] * 100)
    filename = "figures/experiment_3_d%i" % int(distribution[0] * 100)

    tabrow = gather_and_present(title, filename, streams, what)
    print(tabrow)
    text_file.write(tabrow + "\n")
text_file.close()

# Compare distributions
print("Label noise")
text_file = open("rows/label_noises.tex", "w")
for flip_y in label_noises:
    streams = {}
    for drift_type in drift_types:
        for random_state in random_states:
            for distribution in ldistributions:
                stream = csm.StreamGenerator(
                    drift_type=drift_type,
                    distribution=distribution,
                    random_state=random_state,
                    flip_y=flip_y,
                )
                streams.update({str(stream): stream})

    title = "%i%% of label noise" % int(flip_y * 100)
    what = "%.0f\\%%" % (flip_y * 100)
    filename = "figures/experiment_3_ln%i" % int(flip_y * 100)

    tabrow = gather_and_present(title, filename, streams, what)
    print(tabrow)
    text_file.write(tabrow + "\n")
text_file.close()
