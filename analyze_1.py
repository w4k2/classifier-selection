"""
Hyperparameters overview
"""
import numpy as np
from scipy.stats import mannwhitneyu, ranksums
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

results = np.load("results/experiment_1.npy")
drift_types = ("sudden", "incremental")
ensemble_sizes = (2, 3, 4, 5, 10, 15, 20)
alphas = (0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
p = 0.05

print(results.shape)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for i, drift_type in enumerate(drift_types):
    print(drift_type)
    subresults = results[i, :, :]
    mean_score = np.mean(subresults, axis=2)

    ind = np.unravel_index(np.argmax(mean_score, axis=None), mean_score.shape)

    ax[i].set_title(drift_type)
    ax[i].imshow(mean_score, cmap="coolwarm", interpolation="nearest")

    idependences = np.zeros((len(ensemble_sizes), len(alphas))).astype(bool)
    for a in range(len(ensemble_sizes)):
        for b in range(len(alphas)):
            test = mannwhitneyu(subresults[ind], subresults[a, b]).pvalue > p
            idependences[a, b] = test
            ax[i].text(
                b,
                a,
                "%.3f" % mean_score[a, b],
                fontsize=12,
                ha="center",
                va="center",
                color="#000000" if test else "#555555",
            )

    print(mean_score, np.max(mean_score), ind)
    print(idependences)

    print(mean_score.shape)

    plt.sca(ax[i])
    plt.xticks(range(len(alphas)), alphas)
    plt.yticks(range(len(ensemble_sizes)), ensemble_sizes)

plt.tight_layout()
plt.savefig("figures/experiment_1.eps")
