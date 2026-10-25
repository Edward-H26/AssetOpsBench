"""CSV helpers shared by the evaluation scripts: a quantity measured under each fold seed becomes its mean, range and
per-seed values, and an interval comes from a bootstrap that resamples whole clusters."""
import csv
import random
import statistics
from pathlib import Path


def seed_summary(values, digits=4):
    """One quantity measured under each fold seed: its mean, minimum, maximum and the values in seed order."""
    def r(v):
        return round(v, digits)
    return {"mean": r(statistics.mean(values)), "min": r(min(values)), "max": r(max(values)),
            "by_seed": " ".join(str(r(v)) for v in values)}


def seed_columns(name, values, digits=4):
    """seed_summary as columns named after the quantity."""
    return {f"{name}_{k}": v for k, v in seed_summary(values, digits).items()}


def cluster_resamples(n_clusters, n_resamples, seed):
    """Index lists of a bootstrap that draws n_clusters whole clusters with replacement, from a fixed seed so that every
    interval reproduces; one set of resamples can serve every row of a table."""
    rng = random.Random(seed)
    return [rng.choices(range(n_clusters), k=n_clusters) for _ in range(n_resamples)]


def percentile_interval(values):
    """95% equal-tailed percentile interval of bootstrap values: their 2.5th and 97.5th percentiles."""
    cuts = statistics.quantiles(values, n=40, method="inclusive")
    return cuts[0], cuts[-1]


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
