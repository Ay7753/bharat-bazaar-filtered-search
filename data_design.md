# Synthetic Data Design

## Scale and vectors

- 50,000 catalog items with 64-dimensional `float32` vectors; fixed seed `42`.
- 12 centres sampled from `N(0, 4.0)`; vectors add Gaussian noise with `std=0.65`.
- There are 20 categories but only 12 clusters by design. Categories share family clusters; clusters 10 and 11 are cross-category premium and budget cohorts. This deliberately prevents a one-category/one-cluster shortcut in the planner.

| Cluster | Assignment |
| ---: | --- |
| 0 | Tamil Nadu silk sarees |
| 1 | Sarees from West Bengal, Assam, and Odisha |
| 2 | Sarees from other non-tiny zones |
| 3 | Kurtas, ethnic wear, menswear, kidswear |
| 4 | Footwear, sports/outdoors |
| 5 | Jewelry, handbags, watches |
| 6 | Home decor, kitchenware, bedding, toys |
| 7 | Groceries, books |
| 8 | Electronics, mobile accessories |
| 9 | Beauty, personal care |
| 10 | All Meghalaya items (deliberate small-zone association) |
| 11 | Budget variants across categories |

## Zone distribution

The first four are the large metro markets. All counts sum to 50,000; zones with fewer than 50 items are below 0.1%.

| Zone | Items | Share |
| --- | ---: | ---: |
| Delhi NCR | 10,000 | 20.00% |
| Maharashtra | 9,000 | 18.00% |
| Karnataka | 8,000 | 16.00% |
| Tamil Nadu | 7,000 | 14.00% |
| Telangana | 3,505 | 7.01% |
| West Bengal | 2,400 | 4.80% |
| Uttar Pradesh | 2,000 | 4.00% |
| Rajasthan | 1,500 | 3.00% |
| Kerala | 1,200 | 2.40% |
| Gujarat | 1,000 | 2.00% |
| Andhra Pradesh | 900 | 1.80% |
| Madhya Pradesh | 800 | 1.60% |
| Punjab | 500 | 1.00% |
| Haryana | 400 | 0.80% |
| Bihar | 350 | 0.70% |
| Odisha | 300 | 0.60% |
| Assam | 250 | 0.50% |
| Jharkhand | 200 | 0.40% |
| Chhattisgarh | 150 | 0.30% |
| Himachal Pradesh | 100 | 0.20% |
| Uttarakhand | 100 | 0.20% |
| Goa | 80 | 0.16% |
| Tripura | 60 | 0.12% |
| Meghalaya | 45 | 0.09% |
| Manipur | 40 | 0.08% |
| Nagaland | 35 | 0.07% |
| Sikkim | 30 | 0.06% |
| Arunachal Pradesh | 25 | 0.05% |
| Mizoram | 20 | 0.04% |
| Ladakh | 10 | 0.02% |

Meghalaya is the small-zone stress case: all 45 rows map to cluster 10, regardless of category. The six other under-50 zones use category-driven cluster assignments and are not assigned a dedicated vector cluster.

## Category mix and regional correlation

There are 20 categories at 5% target share each (2,500 items in expectation): sarees, kurtas, footwear, jewelry, home decor, groceries, electronics, beauty, ethnic wear, menswear, kidswear, handbags, watches, kitchenware, bedding, toys, books, sports/outdoors, mobile accessories, and personal care.

Tamil Nadu is set to 25% sarees; probabilities in other zones are adjusted so the weighted overall saree share remains 5% in expectation. Tamil Nadu sarees prefer cluster 0 (80%); sarees in West Bengal/Assam/Odisha prefer cluster 1 (65%); sarees in other non-tiny zones prefer cluster 2 (68%). Non-saree categories primarily use their shared family cluster, with 10% assigned to budget cluster 11. A separate 12% price-only premium uplift is independent of vector cluster. Meghalaya's cluster-10 override is intentional and exists for the post-filtering experiment.

## Price distributions

Prices are log-normal in INR, clipped to min/max. Columns below are median, log-space standard deviation, minimum, and maximum. A random 12% premium uplift multiplies price by 1.8; budget cluster 11 multiplies price by 0.65. Tamil Nadu sarees have an additional 4x multiplier, creating a severe price/category/zone dependence trap: a planner assuming price is independent of zone given category will misestimate the 10,000-20,000 INR saree slice.

| Category | Median | Log std. dev. | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Sarees | 2,800 | 0.80 | 300 | 50,000 |
| Kurtas | 900 | 0.65 | 200 | 15,000 |
| Footwear | 1,600 | 0.70 | 250 | 20,000 |
| Jewelry | 2,200 | 1.00 | 100 | 100,000 |
| Home decor | 1,400 | 0.80 | 100 | 30,000 |
| Groceries | 450 | 0.45 | 20 | 8,000 |
| Electronics | 12,000 | 0.85 | 500 | 200,000 |
| Beauty | 700 | 0.65 | 50 | 12,000 |
| Ethnic wear | 1,600 | 0.70 | 250 | 25,000 |
| Menswear | 1,400 | 0.70 | 200 | 30,000 |
| Kidswear | 800 | 0.65 | 150 | 12,000 |
| Handbags | 1,800 | 0.90 | 200 | 40,000 |
| Watches | 2,500 | 0.90 | 250 | 100,000 |
| Kitchenware | 900 | 0.70 | 50 | 20,000 |
| Bedding | 1,800 | 0.80 | 200 | 30,000 |
| Toys | 700 | 0.70 | 50 | 15,000 |
| Books | 450 | 0.60 | 30 | 8,000 |
| Sports/outdoors | 1,800 | 0.80 | 200 | 40,000 |
| Mobile accessories | 650 | 0.65 | 50 | 20,000 |
| Personal care | 500 | 0.55 | 30 | 10,000 |

## Selectivity examples

The seeded run constructs approximate planner buckets using exact metadata filters:

| Bucket | Example filter | Expected survivors |
| --- | --- | ---: |
| 100% | No filter | 50,000 |
| 10% | Category is sarees or kurtas | About 5,000 |
| 1% | Electronics and price between its 20th and 40th percentiles | About 500 |
| 0.1% | Zone is Meghalaya | 45 (0.09%) |
| A handful | Ladakh and price between the 3rd and 6th sorted prices | 4 |

## Files and checks

Run `python generate_data.py` from the repository root to regenerate ignored `data/vectors.npy` and `data/metadata.csv`. The CSV rows align with vector rows and contain `item_id`, `zone`, `category`, `cluster_id`, and `price_inr`. The baseline uses Euclidean distance after exact metadata filtering.