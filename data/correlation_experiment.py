import csv
from pathlib import Path

import numpy as np


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    vectors = np.load(data_dir / "vectors.npy", mmap_mode="r")
    with (data_dir / "metadata.csv").open(encoding="utf-8", newline="") as metadata_file:
        metadata = list(csv.DictReader(metadata_file))

    correlated_index = next(
        index
        for index, row in enumerate(metadata)
        if row["zone"] == "Meghalaya" and row["cluster_id"] == "10"
    )
    other_cluster_index = next(
        index for index, row in enumerate(metadata) if row["cluster_id"] == "8"
    )

    def count_zone_survivors(query_index: int) -> int:
        distances = np.linalg.norm(vectors - vectors[query_index], axis=1)
        top_indices = np.argpartition(distances, 99)[:100]
        return sum(metadata[int(index)]["zone"] == "Meghalaya" for index in top_indices)

    correlated_survivors = count_zone_survivors(correlated_index)
    other_cluster_survivors = count_zone_survivors(other_cluster_index)
    print("Filter: zone=Meghalaya (45 of 50,000 items)")
    print(
        f"Cluster 10 query ({metadata[correlated_index]['item_id']}): "
        f"{correlated_survivors}/100 unfiltered top-100 survive"
    )
    print(
        f"Cluster 8 query ({metadata[other_cluster_index]['item_id']}): "
        f"{other_cluster_survivors}/100 unfiltered top-100 survive"
    )
    if correlated_survivors <= other_cluster_survivors:
        raise AssertionError("small-zone correlation did not change post-filter survival")


if __name__ == "__main__":
    main()