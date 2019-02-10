"""
Hyperparameters overview.

Ustalamy typowy strumien zawierający problem o dużej skali niezbalansowania
(1:9) i szumie na poziomie udziału klasy mniejszościowej (.1). Testujemy
uśrednioną miarę i zależność statystyczną przebiegów dla różnych wielkości
komitetów i różnej wartości parametru odcięcia klasyfikatorów o przestarzałej
granicy decyzyjnej. Testujemy osobno strumień z dryfami nagłymi i
inkrementalnymi.

Oczekiwania:
- powiększanie komitetu początkowo stabilizuje jakość, ale z czasem degraduje
zdolność do reakcji na dryf koncepcji,
- zwiększanie progu odcięcia początkowo kompensuje spowolnienie reakcji na
dryf, ale z czasem wpływa negatywnie na stabilność jakości.
- obie wartości powinny być relatywnie niskie.
- czas trwania badań to około pół godziny.
"""

import csm
import numpy as np
from tqdm import tqdm

drift_types = ("sudden", "incremental")
ensemble_sizes = (2, 3, 4, 5, 10, 15, 20)
alphas = (0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)

results = np.zeros((len(drift_types), len(ensemble_sizes), len(alphas), 199))

for i, drift_type in tqdm(enumerate(drift_types), ascii=True, total=len(drift_types)):
    stream = csm.StreamGenerator(
        drift_type=drift_type, distribution=[0.1, 0.9], random_state=844, flip_y=0.1
    )
    for j, ensemble_size in tqdm(
        enumerate(ensemble_sizes), ascii=True, total=len(ensemble_sizes)
    ):
        for k, alpha in tqdm(enumerate(alphas), ascii=True, total=len(alphas)):
            clf = csm.MDE(alpha=alpha, ensemble_size=ensemble_size)

            learner = csm.TestAndTrain(stream, clf)
            learner.run()
            stream.reset()

            results[i, j, k, :] = np.array(learner.scores)
np.save("results/experiment_1", results)
