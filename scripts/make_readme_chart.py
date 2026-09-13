"""Render docs/churn-by-contract.png from the committed sample_data/customer_churn.csv.

    python scripts/make_readme_chart.py

This is the INPUT data, not pipeline output: churn rate per contract type after
the two cleaning steps the README describes (drop exact duplicates, fold the
three casings of `contract_type` into one). No model and no LLM involved.
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "sample_data" / "customer_churn.csv").drop_duplicates()
df["contract_type"] = df["contract_type"].str.strip().str.lower()
g = df.groupby("contract_type")["churned"].agg(["mean", "size"]).sort_values("mean", ascending=False)

fig, ax = plt.subplots(figsize=(8, 3.6), dpi=110)
bars = ax.bar(g.index, g["mean"], color="#4f46e5")
for b, (rate, n) in zip(bars, g.itertuples(index=False)):
    ax.text(b.get_x() + b.get_width() / 2, rate + 0.005, f"{rate:.1%}  (n={n})", ha="center", va="bottom")
ax.axhline(df["churned"].mean(), color="#9ca3af", ls="--", lw=1)
ax.text(len(g) - 0.5, df["churned"].mean() + 0.005, f"overall {df['churned'].mean():.1%}", ha="right", color="#6b7280")
ax.set_title(f"Churn rate by contract type - sample_data/customer_churn.csv ({len(df)} rows after dedup)")
ax.set_ylabel("churn rate")
ax.set_ylim(0, g["mean"].max() * 1.15)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = ROOT / "docs" / "churn-by-contract.png"
fig.savefig(out)
print(f"wrote {out}")
