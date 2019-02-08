from StreamGenerator import StreamGenerator


def streams():
    # Static parameters
    d = [0.2, 0.8]
    n_drifts = 3

    # Variables
    label_noises = [0, 0.1, 0.2, 0.3]
    drift_types = ["incremental", "sudden"]
    rs = 0

    # Prepare streams
    streams = {}
    for drift_type in drift_types:
        rs += 1
        for flip_y in label_noises:
            stream = StreamGenerator(
                drift_type=drift_type,
                flip_y=flip_y,
                random_state=rs,
                distribution=d,
                n_drifts=n_drifts,
            )
            streams.update({str(stream): stream})

    return streams
