import csv
from pathlib import Path

import numpy as np


SEED = 42
VECTOR_DIMENSION = 64

ZONE_COUNTS = {
    "Delhi NCR": 10_000,
    "Maharashtra": 9_000,
    "Karnataka": 8_000,
    "Tamil Nadu": 7_000,
    "Telangana": 5_000,
    "West Bengal": 4_000,
    "Uttar Pradesh": 3_000,
    "Rajasthan": 2_000,
    "Kerala": 1_000,
    "Assam": 750,
    "Odisha": 160,
    "Sikkim": 40,
    "Arunachal Pradesh": 30,
    "Ladakh": 20,
}

CATEGORY_SHARES = {
    "sarees": 0.20,
    "kurtas": 0.18,
    "footwear": 0.14,
    "jewelry": 0.12,
    "home_decor": 0.10,
    "groceries": 0.10,
    "electronics": 0.08,
    "beauty": 0.08,
}

CATEGORY_CLUSTERS = {
    "kurtas": 3,
    "footwear": 4,
    "jewelry": 5,
    "home_decor": 6,
    "groceries": 7,
    "electronics": 8,
    "beauty": 9,
}

PRICE_DISTRIBUTIONS = {
    "sarees": (2_800, 0.80, 300, 50_000),
    "kurtas": (900, 0.65, 200, 15_000),
    "footwear": (1_600, 0.70, 250, 20_000),
    "jewelry": (2_200, 1.00, 100, 100_000),
    "home_decor": (1_400, 0.80, 100, 30_000),
    "groceries": (450, 0.45, 20, 8_000),
    "electronics": (12_000, 0.85, 500, 200_000),
    "beauty": (700, 0.65, 50, 12_000),
}

TINY_ZONES = {"Sikkim", "Arunachal Pradesh", "Ladakh"}
EASTERN_ZONES = {"West Bengal", "Assam", "Odisha"}


def category_probabilities(zone: str) -> np.ndarray:
    shares = np.array(list(CATEGORY_SHARES.values()), dtype=np.float64)
    if zone == "Tamil Nadu":
        probabilities = np.empty_like(shares)
        probabilities[0] = 0.50
        probabilities[1:] = 0.50 * shares[1:] / shares[1:].sum()
        return probabilities

    tamil_nadu_share = ZONE_COUNTS["Tamil Nadu"] / sum(ZONE_COUNTS.values())
    probabilities = (shares - tamil_nadu_share * category_probabilities("Tamil Nadu")) / (1.0 - tamil_nadu_share)
    return probabilities


def sample_clusters(rng: np.random.Generator, zones: np.ndarray, categories: np.ndarray) -> np.ndarray:
    cluster_ids = np.empty(len(zones), dtype=np.int8)
    saree_clusters = np.array([0, 1, 2, 10, 11], dtype=np.int8)

    for zone in np.unique(zones):
        zone_mask = zones == zone
        saree_mask = zone_mask & (categories == "sarees")
        if zone == "Tamil Nadu":
            probabilities = [0.75, 0.05, 0.05, 0.05, 0.10]
        elif zone in EASTERN_ZONES:
            probabilities = [0.05, 0.60, 0.10, 0.05, 0.20]
        elif zone in TINY_ZONES:
            probabilities = [0.20, 0.20, 0.30, 0.10, 0.20]
        else:
            probabilities = [0.05, 0.05, 0.60, 0.10, 0.20]
        cluster_ids[saree_mask] = rng.choice(saree_clusters, size=saree_mask.sum(), p=probabilities)

        for category, main_cluster in CATEGORY_CLUSTERS.items():
            mask = zone_mask & (categories == category)
            cluster_ids[mask] = rng.choice(
                [main_cluster, 10, 11], size=mask.sum(), p=[0.78, 0.12, 0.10]
            )

    return cluster_ids


def generate_data() -> tuple[np.ndarray, list[dict[str, object]]]:
    item_count = sum(ZONE_COUNTS.values())
    rng = np.random.default_rng(SEED)
    zones = np.concatenate(
        [np.full(count, zone, dtype="U20") for zone, count in ZONE_COUNTS.items()]
    )
    rng.shuffle(zones)

    categories = np.empty(item_count, dtype="U20")
    category_names = list(CATEGORY_SHARES)
    for zone in ZONE_COUNTS:
        mask = zones == zone
        categories[mask] = rng.choice(
            category_names,
            size=mask.sum(),
            p=category_probabilities(zone),
        )

    cluster_ids = sample_clusters(rng, zones, categories)
    centres = rng.normal(0.0, 4.0, size=(12, VECTOR_DIMENSION)).astype(np.float32)
    vectors = centres[cluster_ids] + rng.normal(
        0.0, 0.65, size=(item_count, VECTOR_DIMENSION)
    ).astype(np.float32)

    prices = np.empty(item_count, dtype=np.float64)
    for category, (median, log_std, minimum, maximum) in PRICE_DISTRIBUTIONS.items():
        mask = categories == category
        samples = rng.lognormal(np.log(median), log_std, size=mask.sum())
        samples *= np.where(cluster_ids[mask] == 10, 1.8, 1.0)
        samples *= np.where(cluster_ids[mask] == 11, 0.65, 1.0)
        prices[mask] = np.clip(samples, minimum, maximum)

    metadata = [
        {
            "item_id": f"item-{index:05d}",
            "zone": str(zones[index]),
            "category": str(categories[index]),
            "cluster_id": int(cluster_ids[index]),
            "price_inr": round(float(prices[index]), 2),
        }
        for index in range(item_count)
    ]
    return vectors.astype(np.float32, copy=False), metadata


def main() -> None:
    vectors, metadata = generate_data()
    output_dir = Path(__file__).resolve().parent
    np.save(output_dir / "vectors.npy", vectors)

    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

    print(f"Generated {len(metadata):,} items with {vectors.shape[1]}-D vectors (seed={SEED}).")
    print("Items per zone:")
    for zone, expected_count in ZONE_COUNTS.items():
        actual_count = sum(item["zone"] == zone for item in metadata)
        print(f"  {zone:20} {actual_count:5,} ({actual_count / len(metadata):6.2%}; expected {expected_count:,})")

    print("Items per category:")
    for category in CATEGORY_SHARES:
        count = sum(item["category"] == category for item in metadata)
        print(f"  {category:12} {count:5,} ({count / len(metadata):6.2%}; target {CATEGORY_SHARES[category]:.0%})")

    tamil_nadu_items = [item for item in metadata if item["zone"] == "Tamil Nadu"]
    tamil_nadu_sarees = sum(item["category"] == "sarees" for item in tamil_nadu_items)
    print(
        f"Tamil Nadu sarees: {tamil_nadu_sarees:,}/{len(tamil_nadu_items):,} "
        f"({tamil_nadu_sarees / len(tamil_nadu_items):.1%}; overall saree rate "
        f"{CATEGORY_SHARES['sarees']:.0%})"
    )


if __name__ == "__main__":
    main()