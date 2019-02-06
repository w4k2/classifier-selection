from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

DRIFT_TYPES = ("sudden", "incremental")


class StreamGenerator:
    def __init__(
        self,
        chunk_size=500,
        n_chunks=200,
        n_features=8,
        distribution=[0.5, 0.5],
        n_drifts=4,
        drift="incremental",
    ):
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        self.distribution = distribution
        self.n_features = n_features
        self.n_drifts = n_drifts
        self.drift = drift
        self.n_classes = len(self.distribution)
        self.is_dry = False
        self.is_prepared = False

        self.stream_length = self.chunk_size * self.n_chunks
        self.n_concepts = self.n_drifts + 2
        self.instances_generated = 0
        self.chunks_generated = 0
        self.samples_per_concept = int(self.stream_length / (self.n_concepts - 1))

        self.classes = np.array([0, 1])

    def prepare(self):
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
        self.concept_usages = np.zeros(self.n_concepts).astype(int)

        # Establish dominant concept for every chunk
        self.concept_dominances = np.linspace(
            0, self.n_drifts + 1, self.n_chunks + 1
        ).astype(int)[:-1]

        # Prepare usage curves
        if self.drift == "incremental":
            # Incremental drift
            self.a = np.round(
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

        if self.drift == "sudden":
            self.a = (self.concept_dominances * self.chunk_size) % (self.chunk_size * 2)

        # print(self.a)
        # print(self.b)
        # print(np.unique(self.b, return_counts=True))
        # print(self.n_concepts)

        # plt.plot(self.a)
        # plt.plot(self.b * 100)
        # plt.savefig("foo.png")

    def get_chunk(self):
        if not self.is_prepared:
            self.prepare()
            self.is_prepared = True

        dominant = self.concept_dominances[self.chunks_generated]
        first, second = (dominant, dominant + 1)

        if self.drift == "sudden":
            if dominant % 2 == 0:
                first, second = (dominant + 1, dominant)
        amount = self.a[self.chunks_generated]

        proportion = np.array([amount, self.chunk_size - amount])

        address_a = self.concept_usages[[first, second]]

        self.concept_usages[[first, second]] += proportion
        address_b = self.concept_usages[[first, second]]

        # print(address_a, address_b)

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

        """
        # print(X.shape, y.shape)
        # print(y)

        print(
            "Chunk",
            self.chunks_generated,
            proportion,
            first,
            second,
            self.concept_usages,
            np.sum(self.concept_usages),
            self.concepts[0][0].shape,
        )
        """

        self.chunks_generated += 1
        if self.chunks_generated == self.n_chunks:
            self.is_dry = True

        return X, y
