"""
Hyperparameters overview
"""
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)

results = np.load("results/experiment_1.npy")
drift_types = ("sudden", "incremental")
ensemble_sizes = (2,3,4,5,10,15,20)
alphas = (0,.05,.1,.15,.2,.25,.3)
p = .05

print(results.shape)
fig, ax =plt.subplots(1,2, figsize=(12,6))
for i, drift_type in enumerate(drift_types):
    print(drift_type)
    subresults = results[i,:,:]
    mean_score = np.mean(subresults, axis=2)

    ind = np.unravel_index(np.argmax(mean_score, axis=None), mean_score.shape)

    ax[i].set_title(drift_type)
    ax[i].imshow(mean_score, cmap='coolwarm', interpolation='nearest')


    idependences = np.zeros((len(ensemble_sizes), len(alphas))).astype(bool)
    for a in range(len(ensemble_sizes)):
        for b in range(len(alphas)):
            test = mannwhitneyu(subresults[ind], subresults[a,b]).pvalue > p
            idependences[a,b] = test
            ax[i].text(b, a, "%.3f" % mean_score[a,b], fontsize=12, ha='center', va='center', alpha=1 if  test else .5)

    print(mean_score, np.max(mean_score), ind)
    print(idependences)

    print(mean_score.shape)



    plt.sca(ax[i])
    plt.xticks(range(len(alphas)), alphas)
    plt.yticks(range(len(ensemble_sizes)), ensemble_sizes)

plt.savefig("foo.png")
