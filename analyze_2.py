"""
Hyperparameters overview
"""
import numpy as np
import matplotlib.pyplot as plt
import helper as h
import csm

np.set_printoptions(precision=3)
p = 0.05


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
score_points = list(range(chunk_size, chunk_size * n_chunks, chunk_size))


def gather_and_present(title, filename, streams, what):
    results_hypercube = np.zeros((len(streams), len(clfs), n_chunks - 1))
    for i, stream_n in enumerate(streams):
        results = np.load("results/experiment_2/%s.npy" % stream_n)
        results_hypercube[i] = results

    overall = np.mean(results_hypercube, axis=0)

    plt.figure(figsize=(8, 4))
    plt.ylim((0.5, 1))
    plt.xlim(0, 99500)
    plt.xlabel("Instances processed", fontsize=12)
    plt.ylabel("Balanced accuracy", fontsize=12)

    plt.yticks(
        [0.5, 0.6, 0.7, 0.8, 0.9, 1],
        ["50%", "60%", "70%", "80%", "90%", "100%"],
        fontsize=12,
    )

    xcoords = [16666 * i for i in range(1, 6)]
    for xc in xcoords:
        plt.axvline(x=xc, c="#EECCCC", ls=":", lw=1)

    plt.xticks(
        [0, 25000, 50000, 75000, 100000],
        ["0", "25k", "50k", "75k", "100k"],
        fontsize=12,
    )

    for y in np.linspace(0.6, 0.9, 4):
        plt.plot(
            range(0, 100000),
            [y] * len(range(0, 100000)),
            "--",
            lw=0.5,
            color="#BBBBBB",
            alpha=0.3,
        )

    plt.tick_params(
        axis="both",
        which="both",
        bottom="off",
        top="off",
        labelbottom="on",
        left="off",
        right="off",
        labelleft="on",
    )

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    for j, clfn in enumerate(clfs):
        clf = clfs[clfn]
        plt.plot(score_points, overall[j], label=clfn)

    plt.legend(loc=9, ncol=6, columnspacing=1, frameon=False)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    # plt.savefig(filename + ".png")
    plt.savefig(filename + ".eps")

    a = np.swapaxes(results_hypercube, 1, 0)
    res = np.reshape(a, (6, -1))

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
    filename = "figures/experiment_2_%s" % drift_type
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
    filename = "figures/experiment_2_d%i" % int(distribution[0] * 100)

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
    filename = "figures/experiment_2_ln%i" % int(flip_y * 100)

    tabrow = gather_and_present(title, filename, streams, what)
    print(tabrow)
    text_file.write(tabrow + "\n")
text_file.close()
