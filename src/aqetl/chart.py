"""The committed chart: 30-day PM2.5 trend and per-station anomaly rate.

Rendered by CI on a fresh checkout, so missing data draws a note instead of
raising: a chart step that fails the workflow would stop the data commit.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot: the runner has no display

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from . import report, standards  # noqa: E402
from .config import ATTRIBUTION, CITIES  # noqa: E402

# Okabe-Ito: distinguishable to colour-blind readers and high contrast on both
# GitHub themes. Explicit hex because the default cycle's yellow vanishes on light.
CITY_COLOURS = {"hanoi": "#0072B2", "danang": "#D55E00", "hcmc": "#009E73"}
BENCH_COLOUR = "#B2182B"
WHO_COLOUR = "#5A5A5A"
BAR_COLOUR = "#52527A"
MISSING_COLOUR = "#C2C2CE"

DISPLAY = {city.key: city.name for city in CITIES}
MIN_HOURS = 18


def _top(ax: plt.Axes, daily: pl.DataFrame, today: date) -> None:
    pm = (
        daily.filter((pl.col("parameter") == "pm25") & (pl.col("n_hours") >= MIN_HOURS))
        .sort("date")
    )
    plotted = [
        key for key in (c.key for c in CITIES)
        if not pm.filter(pl.col("city") == key).is_empty()
    ]
    for key in plotted:
        frame = pm.filter(pl.col("city") == key)
        ax.plot(
            frame["date"].to_list(),
            frame["mean_ugm3"].to_list(),
            color=CITY_COLOURS[key],
            linewidth=1.8,
            label=DISPLAY[key],
        )
    # Cities with no reporting station must be stated, not silently missing:
    # Da Nang has no OpenAQ location within 25 km at all.
    absent = [DISPLAY[key] for key in (c.key for c in CITIES) if key not in plotted]
    if absent:
        ax.annotate(
            "No station reported 24 h data: " + ", ".join(absent),
            xy=(0.0, -0.16), xycoords="axes fraction",
            fontsize=7.5, color="#8A5A00", ha="left",
        )
    ax.axhline(
        standards.qcvn_24h("pm25", today),
        color=BENCH_COLOUR,
        linestyle="--",
        linewidth=1.1,
        label=f"QCVN 05:2023 24h ({standards.qcvn_24h('pm25', today):g} µg/m³)",
    )
    who = standards.who_24h("pm25")
    ax.axhline(
        who,
        color=WHO_COLOUR,
        linestyle=":",
        linewidth=1.1,
        label=f"WHO 2021 24h ({who:g} µg/m³)",
    )
    ax.set_title("Daily mean PM2.5, last 30 days", loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("µg/m³")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    # Headroom so the legend sits inside the axes instead of over a line.
    ax.set_ylim(top=max(ax.get_ylim()[1], standards.qcvn_24h("pm25", today) or 0) * 1.35)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    # 30 daily ticks collide at this width; label every third day.
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _bottom(ax: plt.Axes, qc: pl.DataFrame) -> None:
    """Two bars per station: received-but-dubious, and not received at all.

    They are drawn side by side because collapsing them hid the point: a station
    that was offline is not a station that was lying.
    """
    top = (
        qc.filter(pl.col("n_received") > 0)
        .sort("n_received", descending=True)
        .head(12)
        .with_columns(
            (
                pl.col("station_name").fill_null("unknown").str.slice(0, 26)
                + " · "
                + pl.col("parameter")
            ).alias("label"),
            # A station with zero received hours has no quality to rate; it is
            # plotted by missing_pct alone, and matplotlib cannot take a null.
            pl.col("anomaly_pct").fill_null(0.0),
            pl.col("missing_pct").fill_null(100.0),
        )
        .reverse()
    )
    labels = top["label"].to_list()
    y = range(len(labels))
    ax.barh(
        [v + 0.2 for v in y], top["anomaly_pct"].to_list(), height=0.4,
        color=BAR_COLOUR, label="Flagged reading (% of received hours)",
    )
    ax.barh(
        [v - 0.2 for v in y], top["missing_pct"].to_list(), height=0.4,
        color=MISSING_COLOUR, label="No data (% of expected hours)",
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_title(
        "Data quality by station, last 30 days", loc="left", fontsize=12, fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("Percent")
    ax.set_xlim(0, 105)
    ax.tick_params(axis="y", labelsize=7)
    # Above the axes: at lower-right the labels sat on top of the longest bars.
    ax.legend(
        frameon=False, fontsize=7, ncol=2, loc="lower left",
        bbox_to_anchor=(0, 1.0, 1, 0.1), borderaxespad=0,
    )
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def render(
    root: Path = Path("data"),
    out: Path = Path("charts/aq-30d.png"),
    today: date | None = None,
) -> Path:
    today = today or date.today()
    daily = report.build_daily(root, 30, today)
    qc = report.build_station_qc(root, 30, today)

    fig, (top_ax, bottom_ax) = plt.subplots(
        2, 1, figsize=(11, 8), dpi=150, gridspec_kw={"height_ratios": [3, 2]}
    )
    fig.patch.set_facecolor("white")
    for ax in (top_ax, bottom_ax):
        ax.set_facecolor("white")

    if daily.is_empty() and qc.is_empty():
        top_ax.axis("off")
        bottom_ax.axis("off")
        fig.text(
            0.5,
            0.5,
            "Awaiting first successful run\n\n"
            "This chart is generated from live OpenAQ station data.\n"
            "It stays empty until the pipeline has run with a valid API key.",
            ha="center",
            va="center",
            fontsize=11,
            color="#555555",
            linespacing=1.6,
        )
    else:
        _top(top_ax, daily, today)
        _bottom(bottom_ax, qc)

    fig.text(0.01, 0.005, ATTRIBUTION, fontsize=6.5, color="#666666", ha="left")
    fig.suptitle(
        "Vietnam urban air quality — Hanoi, Da Nang, Ho Chi Minh City",
        fontsize=13,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out
