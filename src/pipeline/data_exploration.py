import warnings

import numpy as np
import pandas as pd
import pymannkendall as mk
import seaborn as sns
from matplotlib import pyplot as plt, ticker
from scipy.stats import kruskal
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import kpss

from utils.constants import TARGETS, CONSUMERS_VARS


def plot_targets(df, sufix):
    fig, ax = plt.subplots(figsize=(8, 5))
    for target in TARGETS:
        ax.plot(df.index, df[target], label=target)

    ax.set_xlabel("Year", fontsize=14, fontweight='bold')
    ax.set_ylabel("Consumption (MWh)", fontsize=14, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.ticklabel_format(style="plain", axis="y")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:,.1f}"))

    plt.xticks(rotation=90)

    ax.grid(True, linestyle="--", alpha=0.6)

    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f"../paper/materials/consumption{sufix}.png")
    plt.show()


def plot_acf_pacf(df_state, state_name, lags=12):
    fig, axes = plt.subplots(nrows=len(TARGETS), ncols=2, figsize=(14, 3 * len(TARGETS)))

    for i, target in enumerate(TARGETS):
        serie = df_state[target].dropna()

        # ACF
        plot_acf(serie, ax=axes[i, 0], lags=lags, alpha=0.05, title=f"ACF - {target} ({state_name})")
        axes[i, 0].set_xlabel("Lag (Months)", fontsize=11, fontweight="bold")
        axes[i, 0].set_ylabel("ACF", fontsize=11, fontweight="bold")
        axes[i, 0].set_xticks(range(0, lags + 1, 6))
        axes[i, 0].grid(True, linestyle="--", alpha=0.6)

        # PACF
        plot_pacf(serie, ax=axes[i, 1], lags=lags, alpha=0.05, method="ywm", title=f"PACF - {target} ({state_name})")
        axes[i, 1].set_xlabel("Lag (Months)", fontsize=11, fontweight="bold")
        axes[i, 1].set_ylabel("PACF", fontsize=11, fontweight="bold")
        axes[i, 1].set_xticks(range(0, lags + 1, 6))
        axes[i, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"../paper/materials/acf_pacf_{state_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_boxplpot(dataframe, filename):
    var_groups = {
        "Electricity Consumption (MWh)": TARGETS,
        "Consumers (Count)": CONSUMERS_VARS,
        "Mean 2m Temperature": sorted([c for c in dataframe.columns if "2 metre temperature_" in c and "_mean" in c]),
        "Mean 2m Dewpoint Temperature": sorted([c for c in dataframe.columns if "2 metre dewpoint temperature_" in c and "_mean" in c]),
        "Mean 10m U Wind": sorted([c for c in dataframe.columns if "10 metre U wind component_" in c and "_mean" in c]),
        "Mean 10m V Wind": sorted([c for c in dataframe.columns if "10 metre V wind component_" in c and "_mean" in c]),
        "Mean Mean Sea Level Pressure": sorted([c for c in dataframe.columns if "Mean sea level pressure_" in c and "_mean" in c]),
        "Mean Surface Pressure": sorted([c for c in dataframe.columns if "Surface pressure_" in c and "_mean" in c]),
        "Mean Total Precipitation": sorted([c for c in dataframe.columns if "Total precipitation_" in c and "_mean" in c]),
        "Mean Total Cloud Cover": sorted([c for c in dataframe.columns if "Total cloud cover_" in c and "_mean" in c]),
    }

    palette = sns.color_palette("tab10", len(var_groups))

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(28, 12))
    axes = axes.flatten()

    for i, ((title, cols), color) in enumerate(zip(var_groups.items(), palette)):
        ax = axes[i]

        simplified_labels = [
            f"{c.split('_')[-2]}_{c.split('_')[-1]}" if any(q in c for q in ["Q1", "Q2", "Q3", "Q4"])
            else c.replace(" Consumers", "")
            for c in cols
        ]

        data = [dataframe[c].dropna().values for c in cols]

        bp = ax.boxplot(
            data,
            tick_labels=simplified_labels,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker="o", markersize=3, alpha=0.4, markerfacecolor=color)
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=9, rotation=90)
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)

    plt.tight_layout()
    plt.savefig(f"../paper/materials/{filename}.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_heatmap_correlation(
    dataframe,
    title="Correlation Matrix",
    figsize=(12, 8),
    fmt=".2f",
):
    numeric_cols = dataframe.select_dtypes(include="number").columns.tolist()

    if TARGETS is not None:
        ordered_targets = [c for c in TARGETS if c in numeric_cols]
        other_cols = [c for c in numeric_cols if c not in ordered_targets]
        ordered_cols = ordered_targets + other_cols
    else:
        ordered_cols = numeric_cols

    # Filter out columns with zero variance to avoid empty NaN correlation rows/columns
    zero_var_cols = [c for c in ordered_cols if dataframe[c].std() == 0]
    if zero_var_cols:
        ordered_cols = [c for c in ordered_cols if c not in zero_var_cols]

    feature_cols = [c for c in ordered_cols if c not in (ordered_targets if TARGETS else [])]
    if not feature_cols and TARGETS and len(ordered_targets) > 0:
        return

    corr_matrix = dataframe[ordered_cols].dropna().corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=fmt,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

def run_hypothesis_tests(df_state, state_name):
    rows = []
    for target in TARGETS:
        serie = df_state[target].dropna()

        # 1. KPSS Test (Stationarity)
        # H0: Series is stationary | H1: Series has unit root (non-stationary)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InterpolationWarning)
            kpss_stat, kpss_p = kpss(serie, regression="c", nlags="auto", result_object=True)
            kpss_concl = "Stationary" if kpss_p >= 0.05 else "Non-Stationary"
            kpss_p_str = "<= 0.01" if kpss_p <= 0.01 else (">= 0.10" if kpss_p >= 0.10 else f"{kpss_p:.4f}")

        # 2. Mann-Kendall Test (Monotonic Trend)
        # H0: No monotonic trend | H1: Presence of monotonic trend
        mk_res = mk.original_test(serie)
        if mk_res.p < 0.05:
            mk_concl = "Upward Trend" if mk_res.Tau > 0 else "Downward Trend"
        else:
            mk_concl = "No Trend"
        mk_p_str = f"{mk_res.p:.2e}" if mk_res.p < 0.001 else f"{mk_res.p:.4f}"

        # 3. Kruskal-Wallis Test (Monthly Seasonality)
        # H0: Medians of 12 months are equal (Non-Seasonal) | H1: At least two months differ (Seasonal)
        monthly_groups = [group.values for _, group in serie.groupby(serie.index.month)]
        kw_stat, kw_p = kruskal(*monthly_groups)
        kw_concl = "Seasonal" if kw_p < 0.05 else "Non-Seasonal"
        kw_p_str = f"{kw_p:.2e}" if kw_p < 0.001 else f"{kw_p:.4f}"

        rows.append({
            "State": state_name,
            "Consumption": target,
            "KPSS Stat": np.round(kpss_stat, 4),
            "KPSS p-value": kpss_p_str,
            "KPSS Conclusion (alpha=0.05)": kpss_concl,
            "MK Tau": np.round(mk_res.Tau, 4),
            "MK Sen Slope": np.round(mk_res.slope, 2),
            "MK p-value": mk_p_str,
            "MK Conclusion (alpha=0.05)": mk_concl,
            "KW Stat": np.round(kw_stat, 4),
            "KW p-value": kw_p_str,
            "KW Conclusion (alpha=0.05)": kw_concl
        })

    df_res = pd.DataFrame(rows)
    df_res = df_res.set_index(["State", "Consumption"])
    return df_res