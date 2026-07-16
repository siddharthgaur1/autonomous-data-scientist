"""Generate the three sample datasets. Reproducible: fixed seed, same bytes every run.

    python sample_data/generate.py

Each dataset carries deliberate defects — missing values, a text column holding
numbers, inconsistent categorical spellings, duplicate rows, outliers — so the
Cleaning Agent has something real to do. The signal is genuine but noisy: a model
should land well short of perfect, because a demo that scores 1.00 is a demo
that's leaking.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
SEED = 42


def _blank(rng: np.random.Generator, series: pd.Series, frac: float) -> pd.Series:
    """Knock out `frac` of the values at random."""
    out = series.copy().astype(object)
    idx = rng.choice(len(out), size=int(len(out) * frac), replace=False)
    out.iloc[idx] = np.nan
    return out


def sales_data(n: int = 500) -> pd.DataFrame:
    """Regression: forecast revenue from spend, staffing and seasonality."""
    rng = np.random.default_rng(SEED)

    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    region = rng.choice(["North", "South", "East", "West"], n, p=[0.3, 0.3, 0.2, 0.2])
    ad_spend = rng.gamma(shape=4.0, scale=550, size=n)
    staff = rng.integers(3, 18, n)
    promo = rng.choice([0, 1], n, p=[0.72, 0.28])
    month = dates.month.to_numpy()

    seasonal = 1800 * np.sin(2 * np.pi * month / 12)
    region_lift = pd.Series(region).map(
        {"North": 2600.0, "South": 1400.0, "East": 700.0, "West": 1900.0}
    ).to_numpy()

    revenue = (
        9000
        + 3.1 * ad_spend
        + 340 * staff
        + 2400 * promo
        + seasonal
        + region_lift
        + rng.normal(0, 2100, n)
    ).round(2)

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "region": region,
            "ad_spend": ad_spend.round(2),
            "staff_count": staff,
            "promo_active": promo,
            "units_sold": (revenue / rng.uniform(45, 65, n)).round().astype(int),
            "revenue": revenue,
        }
    )

    # Messy column: revenue arrives from the finance export as formatted text.
    df["revenue"] = df["revenue"].map(lambda v: f"${v:,.2f}")

    # Inconsistent spellings, the classic hand-entered category problem.
    noisy = rng.choice(len(df), size=60, replace=False)
    df.loc[noisy, "region"] = [
        rng.choice([" north ", "NORTH", "North "]) for _ in noisy
    ]

    df["ad_spend"] = _blank(rng, df["ad_spend"], 0.06)
    df["staff_count"] = _blank(rng, df["staff_count"], 0.04)

    # A few outliers: genuine end-of-quarter blowouts, not corruption.
    spikes = rng.choice(len(df), size=5, replace=False)
    df.loc[spikes, "units_sold"] = df.loc[spikes, "units_sold"] * 9

    return pd.concat([df, df.iloc[:6]], ignore_index=True)  # duplicate rows


def customer_churn(n: int = 800) -> pd.DataFrame:
    """Binary classification: which customers churn."""
    rng = np.random.default_rng(SEED + 1)

    tenure = rng.integers(1, 72, n)
    monthly = rng.uniform(20, 120, n).round(2)
    support = rng.poisson(1.4, n)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.28, 0.17]
    )
    payment = rng.choice(
        ["Credit card", "Bank transfer", "Electronic check", "Mailed check"], n
    )
    has_fibre = rng.choice([0, 1], n, p=[0.45, 0.55])
    satisfaction = np.clip(rng.normal(6.8, 1.9, n), 1, 10).round(1)

    contract_risk = pd.Series(contract).map(
        {"Month-to-month": 1.5, "One year": -0.3, "Two year": -1.2}
    ).to_numpy()

    logit = (
        -1.1
        + contract_risk
        + 0.34 * support
        - 0.031 * tenure
        + 0.013 * monthly
        - 0.28 * satisfaction
        + 0.25 * has_fibre
        + rng.normal(0, 0.85, n)
    )
    churned = (1 / (1 + np.exp(-logit)) > rng.uniform(0, 1, n)).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": (tenure * monthly).round(2),
            "support_tickets": support,
            "contract_type": contract,
            "payment_method": payment,
            "has_fibre": has_fibre,
            "satisfaction_score": satisfaction,
            "churned": churned,
        }
    )

    # Messy column: total_charges is text with commas, and blanks for new accounts.
    df["total_charges"] = df["total_charges"].map(lambda v: f"{v:,.2f}")
    df.loc[df["tenure_months"] <= 1, "total_charges"] = " "

    df["satisfaction_score"] = _blank(rng, df["satisfaction_score"], 0.08)
    df["payment_method"] = _blank(rng, df["payment_method"], 0.03)

    mixed = rng.choice(len(df), size=70, replace=False)
    df.loc[mixed, "contract_type"] = [
        rng.choice(["month-to-month", "MONTH-TO-MONTH", " Month-to-month"])
        for _ in mixed
    ]

    return pd.concat([df, df.iloc[:4]], ignore_index=True)


def stock_prices(n: int = 600) -> pd.DataFrame:
    """Timeseries: next-day close for a single synthetic ticker."""
    rng = np.random.default_rng(SEED + 2)

    dates = pd.bdate_range("2022-01-03", periods=n)
    returns = rng.normal(0.0004, 0.017, n)
    close = 100 * np.exp(np.cumsum(returns))

    volume = (rng.lognormal(13.2, 0.42, n) * (1 + np.abs(returns) * 12)).round()
    high = close * (1 + np.abs(rng.normal(0, 0.009, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.009, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, n))

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "ticker": "SYNTH",
            "open": open_.round(2),
            "high": high.round(2),
            "low": low.round(2),
            "volume": volume.astype(int),
            "close": close.round(2),
        }
    )

    # Messy column: volume from the vendor feed, with 'K'/'M' suffixes.
    def _fmt(v: int) -> str:
        if v >= 1_000_000:
            return f"{v / 1_000_000:.2f}M"
        return f"{v / 1000:.1f}K"

    df["volume"] = df["volume"].map(_fmt)

    # Mixed date formats, as if two exports were concatenated.
    swap = rng.choice(len(df), size=80, replace=False)
    df.loc[swap, "date"] = pd.to_datetime(df.loc[swap, "date"]).dt.strftime("%d/%m/%Y")

    df["open"] = _blank(rng, df["open"], 0.05)
    df["high"] = _blank(rng, df["high"], 0.03)

    return df


def main() -> None:
    for name, frame in {
        "sales_data": sales_data(),
        "customer_churn": customer_churn(),
        "stock_prices": stock_prices(),
    }.items():
        path = OUT / f"{name}.csv"
        frame.to_csv(path, index=False)
        missing = int(frame.isna().sum().sum())
        print(f"{path.name}: {len(frame)} rows x {frame.shape[1]} cols, {missing} missing")


if __name__ == "__main__":
    main()
