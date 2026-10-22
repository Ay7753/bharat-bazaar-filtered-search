# Synthetic Data Design

## Scale and vectors

- 50,000 catalog items, each represented by a 64-dimensional `float32` vector.
- 12 clusters: Tamil Nadu silk sarees; East Indian cotton sarees; North/West designer sarees; kurtas; footwear; jewelry; home decor; groceries; electronics; beauty; premium cross-category items; and budget cross-category items.
- Use seed `42`. Generate one 64-D centre per cluster, then add Gaussian noise (`std=0.65`) to each point around its assigned centre. Centres are sampled with `std=4.0` so clusters remain visible but not perfectly separable.
- Cluster/category mapping: clusters 0-2 map to sarees, 3 to kurtas, 4 to footwear, 5 to jewelry, 6 to home decor, 7 to groceries, 8 to electronics, and 9 to beauty. Clusters 10 and 11 are premium/budget variants used by every category.

## Zone distribution

The four large metro-market zones are Delhi NCR (10,000; 20%), Maharashtra (9,000; 18%), Karnataka (8,000; 16%), and Tamil Nadu (7,000; 14%). Other zones are Telangana (5,000; 10%), West Bengal (4,000; 8%), Uttar Pradesh (3,000; 6%), Rajasthan (2,000; 4%), Kerala (1,000; 2%), Assam (750; 1.5%), and Odisha (160; 0.32%). The intentionally tiny zones are Sikkim (40; 0.08%), Arunachal Pradesh (30; 0.06%), and Ladakh (20; 0.04%). Counts sum to 50,000; each tiny zone is below 0.1%.

## Category distribution and regional correlation

Target overall category shares are sarees 20%, kurtas 18%, footwear 14%, jewelry 12%, home decor 10%, groceries 10%, electronics 8%, and beauty 8%. For Tamil Nadu, assign sarees with probability 50%; distribute the remaining 50% across other categories in proportion to their overall target shares. For the other zones, adjust the category probabilities so the weighted aggregate remains at the target shares in expectation. This creates a strong sarees/Tamil Nadu pair without changing the intended overall mix.

Cluster choice is conditional on category. Tamil Nadu sarees strongly prefer cluster 0 (silk); West Bengal sarees prefer cluster 1 (cotton); other-zone sarees prefer cluster 2 (designer). Other categories primarily use their mapped cluster, with some probability assigned to premium/budget clusters. For tiny zones, cluster selection uses the category-only distribution, independent of zone, to contrast with these regional associations.

## Price distributions

Sample prices from category-specific log-normal distributions in INR, using the listed median and log-space standard deviation, then clip to the listed bounds. Premium and budget clusters further multiply price by 1.8 and 0.65 respectively before clipping.

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

## Files and checks

Save vectors as `data/vectors.npy` and row-aligned metadata as `data/metadata.csv` (`item_id`, `zone`, `category`, `cluster_id`, `price_inr`). The generator prints zone and category counts so the skew can be checked directly. The brute-force baseline uses Euclidean distance over rows passing exact metadata filters.