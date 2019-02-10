"""
Hyperparameters overview
"""

from StreamGenerator import StreamGenerator
from MethodAlternate import MethodAlternate
from TestAndTrain import TestAndTrain
from tqdm import tqdm
import numpy as np

drift_types = ("sudden", "incremental")
ensemble_sizes = (2,3,4,5,10,15,20)
alphas = (0,.05,.1,.15,.2,.25,.3)

results = np.zeros((
    len(drift_types),
    len(ensemble_sizes),
    len(alphas),
    199
))

for i, drift_type in tqdm(enumerate(drift_types),ascii=True, total = len(drift_types)):
    stream = StreamGenerator(
        drift_type=drift_type,
        distribution=[.1, .9],
        random_state=844,
        flip_y=.1,
    )
    for j, ensemble_size in tqdm(enumerate(ensemble_sizes),ascii=True, total = len(ensemble_sizes)):
        for k, alpha in tqdm(enumerate(alphas),ascii=True, total = len(alphas)):
            clf = MethodAlternate(alpha=alpha,
                                  ensemble_size=ensemble_size)

            learner = TestAndTrain(stream, clf)
            learner.run()
            stream.reset()

            results[i,j,k,:] = np.array(learner.scores)
np.save("results/experiment_1", results)
