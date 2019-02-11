"""
Hyperparameters overview
"""
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import helper as h

np.set_printoptions(precision=3)


# Select streams and methods
streams = h.streams()
clfs = h.clfs()

# Stream Variables
drift_types = ["incremental", "sudden"]
distributions = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]
random_states = [1337, 666, 42]
label_noises = [0.0, 0.1, 0.2, 0.3]

# Prepare storage for results
chunk_size = next(iter(streams.values())).chunk_size
n_chunks = next(iter(streams.values())).n_chunks
results_hypercube = np.zeros((len(streams), len(clfs), n_chunks - 1))

for i, stream_n in enumerate(streams):
    results = np.load("results/experiment_2/%s.npy" % stream_n)
    results_hypercube[i] = results

a = np.mean(results_hypercube, axis = 0)

print(a.shape)

score_points = list(range(chunk_size, chunk_size * n_chunks, chunk_size))


plt.figure(figsize=(8, 4))
plt.ylim((0.4, 1))
for j, clfn in enumerate(clfs):
    clf = clfs[clfn]

    plt.plot(score_points, a[j], label=clfn)

plt.legend()
plt.savefig("foo.png")
