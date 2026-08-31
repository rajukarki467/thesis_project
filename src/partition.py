import numpy as np
import pandas as pd


def dirichlet_partition(
    y,
    num_clients,
    alpha,
    seed=42,
    min_samples_per_client=10
):
    """
    Partition dataset indices according
    to a Dirichlet distribution.

    Parameters
    ----------
    y : np.ndarray
        Class labels.

    num_clients : int
        Number of simulated clients.

    alpha : float
        Dirichlet concentration parameter.

    seed : int
        Random seed.

    min_samples_per_client : int
        Minimum number of samples required
        for each client.

    Returns
    -------
    dict
        client_id -> sample indices
    """

    y = np.asarray(y)

    rng = np.random.default_rng(seed)

    classes = np.unique(y)

    while True:

        client_indices = {
            client_id: []
            for client_id in range(num_clients)
        }

        for class_id in classes:

            class_indices = np.where(
                y == class_id
            )[0]

            rng.shuffle(
                class_indices
            )

            proportions = rng.dirichlet(
                np.repeat(
                    alpha,
                    num_clients
                )
            )

            split_points = (
                np.cumsum(
                    proportions
                )
                * len(class_indices)
            ).astype(int)

            split_points[-1] = (
                len(class_indices)
            )

            start = 0

            for client_id, end in enumerate(
                split_points
            ):

                indices = class_indices[
                    start:end
                ]

                client_indices[
                    client_id
                ].extend(
                    indices.tolist()
                )

                start = end

        sizes = [
            len(client_indices[i])
            for i in range(num_clients)
        ]

        if min(sizes) >= min_samples_per_client:
            break

    for client_id in client_indices:

        rng.shuffle(
            client_indices[client_id]
        )

    return client_indices

def client_class_distribution(
    y,
    client_indices
):
    """
    Calculate class percentages
    for each client.
    """

    y = np.asarray(y)

    classes = np.unique(y)

    result = []

    for client_id, indices in client_indices.items():

        client_labels = y[
            indices
        ]

        counts = pd.Series(
            client_labels
        ).value_counts()

        total = len(
            client_labels
        )

        row = {
            "client": client_id
        }

        for class_id in classes:

            count = counts.get(
                class_id,
                0
            )

            row[
                f"class_{class_id}"
            ] = count / total

        result.append(row)

    return pd.DataFrame(result)