from sklearn.datasets import make_classification
import numpy as np

DRIFT_TYPES = ("sudden", "incremental")


class StreamGenerator:
    def __init__(
        self,
        chunk_size=500,
        n_chunks=200,
        n_features=8,
        distribution=[0.5, 0.5],
        n_drifts=4,
        drift_type="incremental",
        random_state=None,
    ):
        # Store stream parameters
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        self.n_drifts = n_drifts
        self.stream_length = self.chunk_size * self.n_chunks

        # Store concept parameters
        self.random_state = random_state
        self.n_concepts = self.n_drifts + 2
        self.distribution = distribution
        self.classes = np.array(range(len(self.distribution)))
        self.n_features = n_features
        self.drift_type = drift_type
        self.n_classes = len(self.distribution)
        self.samples_per_concept = int(self.stream_length / (self.n_concepts - 1))

        # Calculate processing variables
        self.reset()
        self.is_prepared = False

    def reset(self):
        self.is_dry = False
        self.chunks_generated = 0
        self.concept_usages = np.zeros(self.n_concepts).astype(int)

    def prepare(self):
        # Prepare random state
        np.random.seed(self.random_state)

        # Prepare concepts
        self.concepts = [
            make_classification(
                n_samples=self.samples_per_concept,
                n_features=self.n_features,
                n_classes=self.n_classes,
                weights=self.distribution,
                class_sep=0.001,
                shuffle=True,
            )
            for i in range(self.n_concepts)
        ]

        # Establish dominant concept for every chunk
        self.concept_dominances = np.linspace(
            0, self.n_drifts + 1, self.n_chunks + 1
        ).astype(int)[:-1]

        # Prepare usage curves
        if self.drift_type == "incremental":
            # Incremental drift
            self.usage_curve = np.round(
                np.abs(
                    (
                        np.cos(
                            np.linspace(
                                0, np.pi * (self.n_drifts + 1) / 2, self.n_chunks + 1
                            )
                            % (np.pi / 2)
                        )
                        * self.chunk_size
                    )
                )
            ).astype(int)[:-1]

        elif self.drift_type == "sudden":
            self.usage_curve = (self.concept_dominances * self.chunk_size) % (
                self.chunk_size * 2
            )

    def get_chunk(self):
        if not self.is_prepared:
            self.prepare()
            self.is_prepared = True

        dominant = self.concept_dominances[self.chunks_generated]
        first, second = (dominant, dominant + 1)

        if self.drift_type == "sudden":
            if dominant % 2 == 0:
                first, second = (dominant + 1, dominant)
        amount = self.usage_curve[self.chunks_generated]

        proportion = np.array([amount, self.chunk_size - amount])

        address_a = self.concept_usages[[first, second]]

        self.concept_usages[[first, second]] += proportion
        address_b = self.concept_usages[[first, second]]

        X = np.append(
            self.concepts[first][0][address_a[0] : address_b[0]],
            self.concepts[second][0][address_a[1] : address_b[1]],
            axis=0,
        )
        y = np.append(
            self.concepts[first][1][address_a[0] : address_b[0]],
            self.concepts[second][1][address_a[1] : address_b[1]],
            axis=0,
        )

        self.chunks_generated += 1
        if self.chunks_generated == self.n_chunks:
            self.is_dry = True

        return X, y
