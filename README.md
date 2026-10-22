# Bharat Bazaar Filtered Search

Generate the deterministic 50,000-item catalog and its 64-dimensional vectors:

```powershell
python generate_data.py
```

This creates `data/vectors.npy` and `data/metadata.csv`. Both generated files are ignored by Git; regenerate them locally instead of downloading or committing the dataset.

Run the small-zone post-filtering experiment after generation:

```powershell
python data/correlation_experiment.py
```

Run the brute-force baseline tests:

```powershell
python -m unittest discover -s data -p "test_*.py" -v
```