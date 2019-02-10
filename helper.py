from StreamGenerator import StreamGenerator


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
                    stream = StreamGenerator(
                        drift_type=drift_type,
                        distribution=distribution,
                        random_state=random_state,
                        flip_y=flip_y,
                    )
                    streams.update({str(stream): stream})

    return streams
