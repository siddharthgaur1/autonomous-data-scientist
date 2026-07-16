"""Plotly figures for EDA and the report deck.

These are hand-written rather than LLM-generated. Charts land in a slide deck a
human presents, and a model improvising `fig.update_layout` produces a different
look on every run; a single registered template keeps the deck coherent and keeps
the palette honest.

Palette: the validated reference instance (see the project README). Correlations
use the diverging blue<->red pair with a neutral gray midpoint; magnitude uses
the single-hue blue ramp. Every figure is written for the light chart surface,
which is what the .pptx renders on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

#: Fixed slot order — assigned in order, never cycled.
CATEGORICAL = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834"]
SERIES_1 = CATEGORICAL[0]

#: Single hue, light->dark, for magnitude.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#184f95"]

#: Two poles + neutral gray midpoint. Never a hue at the middle.
DIVERGING = [(0.0, "#d03b3b"), (0.5, "#f0efec"), (1.0, "#2a78d6")]

TEMPLATE = "ads"


def register_template() -> None:
    """Register the project's plotly template and make it the default."""
    pio.templates[TEMPLATE] = go.layout.Template(
        layout=go.Layout(
            colorway=CATEGORICAL,
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
                  "size": 13, "color": INK_SECONDARY},
            title={"font": {"size": 17, "color": INK_PRIMARY}, "x": 0.0, "xanchor": "left"},
            xaxis={"gridcolor": GRIDLINE, "linecolor": BASELINE, "zerolinecolor": BASELINE,
                   "tickfont": {"color": INK_MUTED}, "automargin": True},
            yaxis={"gridcolor": GRIDLINE, "linecolor": BASELINE, "zerolinecolor": BASELINE,
                   "tickfont": {"color": INK_MUTED}, "automargin": True},
            margin={"l": 60, "r": 30, "t": 55, "b": 55},
            bargap=0.25,
            showlegend=False,
        )
    )
    pio.templates.default = TEMPLATE


def save_fig(fig: go.Figure, run_dir: Path, name: str) -> Path:
    """Write a figure as PNG for the deck, with an HTML fallback if kaleido is absent.

    Returns whichever file was written. The dashboard is happy with either; the
    report agent needs the PNG and skips the slide if it only got HTML.
    """
    run_dir = Path(run_dir)
    plots = run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    png = plots / f"{name}.png"
    try:
        fig.write_image(str(png), width=1000, height=560, scale=2)
        fig.write_html(str(plots / f"{name}.html"), include_plotlyjs="cdn")
        return png
    except Exception:  # noqa: BLE001 - kaleido missing or no system libs; HTML still useful
        html = plots / f"{name}.html"
        fig.write_html(str(html), include_plotlyjs="cdn")
        return html


def target_distribution(df: pd.DataFrame, target: str) -> go.Figure:
    """Distribution of the target: bars for classes, histogram for a number."""
    series = df[target].dropna()

    if series.dtype == object or series.nunique() <= 10:
        counts = series.value_counts().sort_index()
        fig = go.Figure(
            go.Bar(
                x=[str(i) for i in counts.index],
                y=counts.to_numpy(),
                marker_color=SERIES_1,
                marker_line_width=0,
                text=counts.to_numpy(),
                textposition="outside",
                textfont={"color": INK_SECONDARY},
                hovertemplate=f"{target}=%{{x}}<br>%{{y}} rows<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Target balance: {target}",
            xaxis_title=target,
            yaxis_title="rows",
        )
        return fig

    fig = go.Figure(
        go.Histogram(
            x=series,
            nbinsx=30,
            marker_color=SERIES_1,
            marker_line={"width": 1, "color": SURFACE},
            hovertemplate=f"{target} %{{x}}<br>%{{y}} rows<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Distribution of {target}", xaxis_title=target, yaxis_title="rows"
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, max_cols: int = 12) -> go.Figure | None:
    """Correlation matrix over numeric columns, diverging around zero.

    Returns None when there aren't two numeric columns to correlate.
    """
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        return None

    numeric = numeric.iloc[:, :max_cols]
    corr = numeric.corr(numeric_only=True).round(2)

    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=list(corr.columns),
            y=list(corr.index),
            colorscale=DIVERGING,
            zmid=0,
            zmin=-1,
            zmax=1,
            xgap=2,
            ygap=2,
            colorbar={"title": "r", "outlinewidth": 0, "tickfont": {"color": INK_MUTED}},
            hovertemplate="%{y} vs %{x}<br>r = %{z}<extra></extra>",
        )
    )
    fig.update_layout(title="Correlation between numeric features")
    fig.update_xaxes(showgrid=False, tickangle=-40)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def target_relationship(df: pd.DataFrame, feature: str, target: str) -> go.Figure:
    """How one feature moves with the target — box per class, scatter otherwise."""
    data = df[[feature, target]].dropna()

    if data[target].dtype == object or data[target].nunique() <= 10:
        fig = go.Figure()
        for i, (level, group) in enumerate(data.groupby(target)):
            fig.add_trace(
                go.Box(
                    y=group[feature],
                    name=str(level),
                    marker_color=CATEGORICAL[i % len(CATEGORICAL)],
                    line={"width": 2},
                    boxpoints=False,
                )
            )
        fig.update_layout(
            title=f"{feature} by {target}", xaxis_title=target, yaxis_title=feature
        )
        return fig

    fig = go.Figure(
        go.Scattergl(
            x=data[feature],
            y=data[target],
            mode="markers",
            marker={"size": 8, "color": SERIES_1, "opacity": 0.6,
                    "line": {"width": 2, "color": SURFACE}},
            hovertemplate=f"{feature} %{{x}}<br>{target} %{{y}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{target} vs {feature}", xaxis_title=feature, yaxis_title=target
    )
    return fig


def missingness(df: pd.DataFrame) -> go.Figure | None:
    """Share of missing values per column. None when nothing is missing."""
    share = (df.isna().mean() * 100).round(1)
    share = share[share > 0].sort_values(ascending=False)
    if share.empty:
        return None

    fig = go.Figure(
        go.Bar(
            x=share.to_numpy(),
            y=list(share.index),
            orientation="h",
            marker_color=SERIES_1,
            marker_line_width=0,
            text=[f"{v}%" for v in share],
            textposition="outside",
            textfont={"color": INK_SECONDARY},
            hovertemplate="%{y}<br>%{x}% missing<extra></extra>",
        )
    )
    fig.update_layout(
        title="Missing values before cleaning",
        xaxis_title="% of rows missing",
        yaxis={"autorange": "reversed"},
    )
    return fig


def model_comparison(names: list[str], scores: list[float], metric: str) -> go.Figure:
    """Baseline scores across candidates, best on top."""
    order = np.argsort(scores)
    names = [names[i] for i in order]
    scores = [scores[i] for i in order]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=names,
            orientation="h",
            marker_color=SERIES_1,
            marker_line_width=0,
            text=[f"{s:.3f}" for s in scores],
            textposition="outside",
            textfont={"color": INK_SECONDARY},
            hovertemplate=f"%{{y}}<br>{metric} %{{x:.3f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Candidate models by {metric}", xaxis_title=metric, yaxis_title=""
    )
    return fig


def feature_importance(importances: dict[str, float], top_n: int = 12) -> go.Figure:
    """Top-N importances from the fitted model."""
    items = list(importances.items())[:top_n][::-1]
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=SERIES_1,
            marker_line_width=0,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            textfont={"color": INK_SECONDARY},
            hovertemplate="%{y}<br>importance %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top {len(items)} features", xaxis_title="importance", yaxis_title=""
    )
    return fig
