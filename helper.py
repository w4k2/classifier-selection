import csm


def clfs():
    return {
        "MDE": csm.MDE(decision="min", ensemble_size=3, alpha=0.05),
        "MDEb": csm.MDE(decision="basic", ensemble_size=3, alpha=0.05),
        "KNORAE": csm.DESlibStream(desMethod="KNORAE", ensemble_size=3),
        "KNORAU": csm.DESlibStream(desMethod="KNORAU", ensemble_size=3),
        "Rank": csm.DESlibStream(desMethod="Rank", ensemble_size=3),
        "LCA": csm.DESlibStream(desMethod="LCA", ensemble_size=3),
    }


def streams():
    # Variables
    distributions = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]
    label_noises = [0.0, 0.1, 0.2, 0.3]
    drift_types = ["incremental", "sudden"]
    random_states = [1337, 666, 42]

    # Prepare streams
    streams = {}
    for drift_type in drift_types:
        for distribution in distributions:
            for random_state in random_states:
                for flip_y in label_noises:
                    stream = csm.StreamGenerator(
                        drift_type=drift_type,
                        distribution=distribution,
                        random_state=random_state,
                        flip_y=flip_y,
                    )
                    streams.update({str(stream): stream})

    return streams