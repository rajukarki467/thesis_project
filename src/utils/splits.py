import numpy as np


def split_records(
    record_names,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    """
    Split records into train/validation/test sets.
    """

    if not np.isclose(
        train_ratio
        + val_ratio
        + test_ratio,
        1.0
    ):
        raise ValueError(
            "Split ratios must sum to 1."
        )

    rng = np.random.default_rng(seed)

    records = np.asarray(
        record_names
    ).copy()

    rng.shuffle(records)

    n = len(records)

    train_end = int(
        n * train_ratio
    )

    val_end = train_end + int(
        n * val_ratio
    )

    train_records = records[
        :train_end
    ]

    val_records = records[
        train_end:val_end
    ]

    test_records = records[
        val_end:
    ]

    return (
        train_records.tolist(),
        val_records.tolist(),
        test_records.tolist()
    )