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
    "Telangana": 3_505,
    "West Bengal": 2_400,
    "Uttar Pradesh": 2_000,
    "Rajasthan": 1_500,
    "Kerala": 1_200,
    "Gujarat": 1_000,
    "Andhra Pradesh": 900,
    "Madhya Pradesh": 800,
    "Punjab": 500,
    "Haryana": 400,
    "Bihar": 350,
    "Odisha": 300,
    "Assam": 250,
    "Jharkhand": 200,
    "Chhattisgarh": 150,
    "Himachal Pradesh": 100,
    "Uttarakhand": 100,
    "Goa": 80,
    "Tripura": 60,
    "Meghalaya": 45,
    "Manipur": 40,
    "Nagaland": 35,
    "Sikkim": 30,
    "Arunachal Pradesh": 25,
    "Mizoram": 20,
    "Ladakh": 10,
}

CATEGORY_SHARES = {
    "sarees": 0.05,
    "kurtas": 0.05,
    "footwear": 0.05,
    "jewelry": 0.05,
    "home_decor": 0.05,
    "groceries": 0.05,
    "electronics": 0.05,
    "beauty": 0.05,
    "ethnic_wear": 0.05,
    "menswear": 0.05,
    "kidswear": 0.05,
    "handbags": 0.05,
    "watches": 0.05,
    "kitchenware": 0.05,
    "bedding": 0.05,
    "toys": 0.05,
    "books": 0.05,
    "sports_outdoors": 0.05,
    "mobile_accessories": 0.05,
    "personal_care": 0.05,
}

CATEGORY_CLUSTERS = {
    "kurtas": 3,
    "ethnic_wear": 3,
    "menswear": 3,
    "kidswear": 3,
    "footwear": 4,
    "sports_outdoors": 4,
    "jewelry": 5,
    "handbags": 5,
    "watches": 5,
    "home_decor": 6,
    "kitchenware": 6,
    "bedding": 6,
    "toys": 6,
    "groceries": 7,
    "books": 7,
    "electronics": 8,
    "mobile_accessories": 8,
    "beauty": 9,
    "personal_care": 9,
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
    "ethnic_wear": (1_600, 0.70, 250, 25_000),
    "menswear": (1_400, 0.70, 200, 30_000),
    "kidswear": (800, 0.65, 150, 12_000),
    "handbags": (1_800, 0.90, 200, 40_000),
    "watches": (2_500, 0.90, 250, 100_000),
    "kitchenware": (900, 0.70, 50, 20_000),
    "bedding": (1_800, 0.80, 200, 30_000),
    "toys": (700, 0.70, 50, 15_000),
    "books": (450, 0.60, 30, 8_000),
    "sports_outdoors": (1_800, 0.80, 200, 40_000),
    "mobile_accessories": (650, 0.65, 50, 20_000),
    "personal_care": (500, 0.55, 30, 10_000),
}

TINY_ZONES = {
    "Meghalaya", "Manipur", "Nagaland", "Sikkim", "Arunachal Pradesh", "Mizoram", "Ladakh"
}
EASTERN_ZONES = {"West Bengal", "Assam", "Odisha"}


def category_probabilities(zone: str) -> np.ndarray:
    shares = np.array(list(CATEGORY_SHARES.values()), dtype=np.float64)
    if zone == "Tamil Nadu":
        probabilities = np.empty_like(shares)
        probabilities[0] = 0.25
        probabilities[1:] = 0.75 * shares[1:] / shares[1:].sum()
        return probabilities

    tamil_nadu_share = ZONE_COUNTS["Tamil Nadu"] / sum(ZONE_COUNTS.values())
    probabilities = (shares - tamil_nadu_share * category_probabilities("Tamil Nadu")) / (1.0 - tamil_nadu_share)
    return probabilities


def sample_clusters(rng: np.random.Generator, zones: np.ndarray, categories: np.ndarray) -> np.ndarray:
    cluster_ids = np.empty(len(zones), dtype=np.int8)
    saree_clusters = np.array([0, 1, 2, 11], dtype=np.int8)

    for zone in np.unique(zones):
        zone_mask = zones == zone
        if zone == "Meghalaya":
            cluster_ids[zone_mask] = 10
            continue

        saree_mask = zone_mask & (categories == "sarees")
        if zone == "Tamil Nadu":
            probabilities = [0.80, 0.03, 0.05, 0.12]
        elif zone in EASTERN_ZONES:
            probabilities = [0.05, 0.65, 0.10, 0.20]
        elif zone in TINY_ZONES:
            probabilities = [0.20, 0.20, 0.30, 0.30]
        else:
            probabilities = [0.04, 0.04, 0.68, 0.24]
        cluster_ids[saree_mask] = rng.choice(saree_clusters, size=saree_mask.sum(), p=probabilities)

        for category, main_cluster in CATEGORY_CLUSTERS.items():
            mask = zone_mask & (categories == category)
            cluster_ids[mask] = rng.choice([main_cluster, 11], size=mask.sum(), p=[0.90, 0.10])

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
        premium_price = rng.random(mask.sum()) < 0.12
        samples *= np.where(premium_price, 1.8, 1.0)
        samples *= np.where(cluster_ids[mask] == 11, 0.65, 1.0)
        tamil_nadu_sarees = mask & (zones == "Tamil Nadu")
        samples *= np.where(tamil_nadu_sarees[mask], 4.0, 1.0)
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
    tn_saree_prices = [
        item["price_inr"]
        for item in metadata
        if item["zone"] == "Tamil Nadu" and item["category"] == "sarees"
    ]
    other_saree_prices = [
        item["price_inr"]
        for item in metadata
        if item["zone"] != "Tamil Nadu" and item["category"] == "sarees"
    ]
    price_band = (10_000, 20_000)
    tn_band_rate = sum(price_band[0] <= price <= price_band[1] for price in tn_saree_prices) / len(tn_saree_prices)
    other_band_rate = sum(price_band[0] <= price <= price_band[1] for price in other_saree_prices) / len(other_saree_prices)
    print(
        f"Price trap, sarees INR {price_band[0]:,}-{price_band[1]:,}: "
        f"Tamil Nadu {tn_band_rate:.1%} vs other zones {other_band_rate:.1%}"
    )

    category_prices = np.array(
        [item["price_inr"] for item in metadata if item["category"] == "electronics"]
    )
    lower_price, upper_price = np.quantile(category_prices, [0.20, 0.40])
    one_percent_count = sum(
        item["category"] == "electronics" and lower_price <= item["price_inr"] <= upper_price
        for item in metadata
    )
    ladakh_prices = sorted(item["price_inr"] for item in metadata if item["zone"] == "Ladakh")
    handful_count = sum(ladakh_prices[2] <= price <= ladakh_prices[5] for price in ladakh_prices)
    print("Selectivity examples:")
    print(f"  no filter: {len(metadata):,} (100%)")
    print(f"  category in sarees/kurtas: {sum(item['category'] in {'sarees', 'kurtas'} for item in metadata):,} (~10%)")
    print(f"  electronics price {lower_price:.2f}-{upper_price:.2f}: {one_percent_count:,} (~1%)")
    print(f"  zone=Meghalaya: {ZONE_COUNTS['Meghalaya']} (~0.1%)")
    print(f"  zone=Ladakh and price {ladakh_prices[2]:.2f}-{ladakh_prices[5]:.2f}: {handful_count} (handful)")


if __name__ == "__main__":
    main()