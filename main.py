import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import textwrap
import re
import unicodedata

# -----------------------
# CONFIG
# -----------------------
XLSX_PATH = "data/seeschifffahrt.xlsx"

SHEET_CARGO_TONS = "46331-b06"
SHEET_CONTAINER_TEU = "46331-b13"
SHEET_PASSENGERS = "46331-b16"

CURRENT_VALUE_COL = 1
YOY_PCT_COL = 5

# Set to False if you want to remove "Übrige Regionen" everywhere
INCLUDE_UEBRIGE_REGIONEN = True

Path("output").mkdir(exist_ok=True)


# -----------------------
# STYLE HELPERS
# -----------------------
def economist_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#111111",
        "axes.labelcolor": "#111111",
        "text.color": "#111111",
        "axes.grid": True,
        "grid.color": "#E8E8E8",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False
    })


def wrap_label(s, width=22):
    return "\n".join(textwrap.wrap(str(s), width=width))


def fmt_big(x, pos):
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1_000:.0f}k"
    return f"{x:.0f}"


# -----------------------
# CANONICALIZATION (THE KEY FIX)
# -----------------------
def canonical_name(s) -> str:
    s = "" if pd.isna(s) else str(s)
    s = s.replace("\u00a0", " ")               # NBSP
    s = unicodedata.normalize("NFKC", s)       # unicode normalize
    s = s.replace("–", "-").replace("—", "-")  # normalize dashes
    s = s.strip()
    s = re.sub(r"\s+", " ", s)                 # collapse whitespace
    return s


def bundesland_key(s) -> str:
    return canonical_name(s).lower()


# -----------------------
# READER
# -----------------------
def read_bundesland_table(sheet_name: str, metric_name: str) -> pd.DataFrame:
    df = pd.read_excel(XLSX_PATH, sheet_name=sheet_name, header=None)

    header_row = 3
    df = df.iloc[header_row:].copy()
    df.columns = df.iloc[0]
    df = df.iloc[1:].copy()

    first_col = df.columns[0]
    df = df.rename(columns={first_col: "bundesland"})

    cols = list(df.columns)
    if len(cols) <= max(CURRENT_VALUE_COL, YOY_PCT_COL):
        raise ValueError(f"Unexpected column layout in {sheet_name}. Columns: {cols}")

    df = df[["bundesland", cols[CURRENT_VALUE_COL], cols[YOY_PCT_COL]]].copy()
    df.columns = ["bundesland", "value", "yoy_pct"]

    df["bundesland"] = df["bundesland"].apply(canonical_name)
    df["bundesland_key"] = df["bundesland"].apply(bundesland_key)

    df = df[~df["bundesland"].str.contains("Ende der Tabelle", case=False, na=False)]
    df = df[~df["bundesland"].str.contains("Insgesamt", case=False, na=False)]
    df = df.dropna(subset=["value"])

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["yoy_pct"] = pd.to_numeric(df["yoy_pct"], errors="coerce")
    df = df.dropna(subset=["value"])

    df["metric"] = metric_name
    return df


# -----------------------
# BUILD DATA LONG
# -----------------------
economist_style()

cargo = read_bundesland_table(SHEET_CARGO_TONS, "cargo_1000_tons")
teu = read_bundesland_table(SHEET_CONTAINER_TEU, "container_1000_teu")
pax = read_bundesland_table(SHEET_PASSENGERS, "passengers_count")

data_long = pd.concat([cargo, teu, pax], ignore_index=True)

if not INCLUDE_UEBRIGE_REGIONEN:
    data_long = data_long[~data_long["bundesland"].str.contains("Übrige Regionen", case=False, na=False)]

# HARD DEDUP: one row per (bundesland_key, metric)
# Keep a stable display label using "first" after canonicalization
data_long = (
    data_long.groupby(["bundesland_key", "metric"], as_index=False)
             .agg(
                 bundesland=("bundesland", "first"),
                 value=("value", "mean"),
                 yoy_pct=("yoy_pct", "mean"),
             )
)

# Remove empty/invalid region names (safety)
data_long = data_long[data_long["bundesland_key"].str.len() > 0].copy()

# Save cleaned long data
data_long.to_csv("output/bundesland_activity_long.csv", index=False)

# -----------------------
# MIX / SHARES (GROUP BY KEY)
# -----------------------
data_long["abs_value"] = data_long["value"].abs()

totals = (
    data_long.groupby("bundesland_key", as_index=False)["abs_value"]
             .sum()
             .rename(columns={"abs_value": "total_activity_proxy"})
)

data_long = data_long.merge(totals, on="bundesland_key", how="left")
data_long["share"] = data_long["abs_value"] / (data_long["total_activity_proxy"] + 1e-9)

# HHI by key
hhi = (
    data_long.groupby("bundesland_key")["share"]
    .apply(lambda s: float(np.sum(s**2)))
    .reset_index(name="hhi")
)

# top1/top2 share by key
def top_shares(s: pd.Series) -> pd.Series:
    s = s.sort_values(ascending=False).reset_index(drop=True)
    top1 = float(s.iloc[0]) if len(s) else np.nan
    top2 = float(s.iloc[:2].sum()) if len(s) >= 2 else float(s.sum())
    return pd.Series({"top1_share": top1, "top2_share": top2})

tops = (
    data_long.groupby("bundesland_key")["share"]
    .apply(top_shares)
    .reset_index()
)

# momentum by key
momentum = (
    data_long.groupby("bundesland_key", as_index=False)["yoy_pct"]
             .mean()
             .rename(columns={"yoy_pct": "avg_yoy_pct"})
)

# label table by key (stable display label)
labels = (
    data_long.groupby("bundesland_key", as_index=False)["bundesland"]
             .first()
)

# total proxy by key
tot_activity = totals.copy()

# -----------------------
# METRICS TABLE (MERGE BY KEY ONLY)
# -----------------------
metrics = (
    labels.merge(hhi, on="bundesland_key", how="left")
          .merge(tops, on="bundesland_key", how="left")
          .merge(momentum, on="bundesland_key", how="left")
          .merge(tot_activity, on="bundesland_key", how="left")
)

# Final safety: enforce uniqueness
metrics = metrics.drop_duplicates(subset=["bundesland_key"]).copy()

metrics["avg_yoy_pct"] = metrics["avg_yoy_pct"].fillna(0.0)
metrics["diversification"] = 1 - metrics["hhi"]

minv, maxv = metrics["diversification"].min(), metrics["diversification"].max()
metrics["diversification_score"] = 100 * (metrics["diversification"] - minv) / (maxv - minv + 1e-9)

m = np.clip(metrics["avg_yoy_pct"], -20, 20)
metrics["momentum_score"] = (m + 20) * (100 / 40)

metrics["resilience_score"] = 0.85 * metrics["diversification_score"] + 0.15 * metrics["momentum_score"]
metrics = metrics.sort_values("resilience_score", ascending=False)

metrics.to_csv("output/bundesland_resilience_metrics.csv", index=False)

# -----------------------
# PLOT DATA
# -----------------------
N = min(10, len(metrics))
top_res = metrics.head(N).sort_values("resilience_score", ascending=True)
top_dep = metrics.sort_values("hhi", ascending=False).head(N).sort_values("hhi", ascending=True)

# Mix pivot by KEY (prevents duplicate index)
mix = data_long.copy()
mix["share_pct"] = 100 * mix["share"]

pivot = (
    mix.pivot_table(index="bundesland_key", columns="metric", values="share_pct", aggfunc="sum")
       .fillna(0)
)

# choose TopK by resilience (using keys)
topK = min(8, len(metrics))
top_keys = metrics.head(topK)["bundesland_key"].tolist()
pivot = pivot.loc[[k for k in top_keys if k in pivot.index]]

# map keys -> display label for y-axis
key_to_name = dict(zip(metrics["bundesland_key"], metrics["bundesland"]))
pivot_names = [key_to_name.get(k, k) for k in pivot.index]

# -----------------------
# FIGURE LAYOUT
# -----------------------
fig = plt.figure(figsize=(12.5, 15), dpi=240)
fig.subplots_adjust(left=0.20, right=0.97, top=0.90, bottom=0.06, hspace=0.55, wspace=0.35)
gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.55, 1.25])

fig.text(0.20, 0.965, "German seaport regions: resilience vs concentration",
         ha="left", va="top", fontsize=18, fontweight="bold")
fig.text(0.20, 0.942, "Mix index across cargo, containers and passengers • Source: Destatis",
         ha="left", va="top", fontsize=11, color="#444444")

# Panel 1
ax1 = fig.add_subplot(gs[0, 0])
ax1.barh([wrap_label(x, 24) for x in top_res["bundesland"]], top_res["resilience_score"])
ax1.set_title(f"Most resilient (Top {N})", loc="left", fontweight="bold", pad=6)
ax1.set_xlabel("Score (0–100)")
ax1.grid(axis="x")
ax1.grid(axis="y", visible=False)

# Panel 2
ax2 = fig.add_subplot(gs[0, 1])
ax2.barh([wrap_label(x, 24) for x in top_dep["bundesland"]], top_dep["hhi"])
ax2.set_title(f"Most concentrated (Top {N})", loc="left", fontweight="bold", pad=6)
ax2.set_xlabel("HHI (higher = more dependent)")
ax2.grid(axis="x")
ax2.grid(axis="y", visible=False)

# Panel 3 scatter (LABELS ONCE — because metrics is unique by key)
ax3 = fig.add_subplot(gs[1, :])

x = metrics["hhi"].astype(float)
y = metrics["total_activity_proxy"].astype(float)

ax3.scatter(x, y, s=70, alpha=0.9)
ax3.set_title("Portfolio view: concentration vs activity size", loc="left", fontweight="bold", pad=6)
ax3.set_xlabel("Concentration (HHI)")
ax3.set_ylabel("Activity size (proxy)")
ax3.yaxis.set_major_formatter(FuncFormatter(fmt_big))

x_med = float(np.median(x))
y_med = float(np.median(y))
ax3.axvline(x_med, linewidth=1.0)
ax3.axhline(y_med, linewidth=1.0)

xpad = max(0.02, 0.08 * (x.max() - x.min() + 1e-9))
ypad = 0.10 * (y.max() - y.min() + 1e-9)
ax3.set_xlim(x.min() - xpad, x.max() + xpad)
ax3.set_ylim(max(0, y.min() - ypad), y.max() + ypad)

ax3.text(0.02, 0.92, "Low conc.\nLarge", transform=ax3.transAxes, fontsize=10, color="#333333")
ax3.text(0.80, 0.92, "High conc.\nLarge", transform=ax3.transAxes, fontsize=10, color="#333333")
ax3.text(0.02, 0.08, "Low conc.\nSmall", transform=ax3.transAxes, fontsize=10, color="#333333")
ax3.text(0.80, 0.08, "High conc.\nSmall", transform=ax3.transAxes, fontsize=10, color="#333333")

# label all points (no repeats now)
offsets = [(6, 6), (6, -10), (-10, 6), (-10, -10), (10, 0), (-12, 0)]
for i, r in metrics.reset_index(drop=True).iterrows():
    dx, dy = offsets[i % len(offsets)]
    ax3.annotate(
        r["bundesland"],
        (float(r["hhi"]), float(r["total_activity_proxy"])),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=9.5,
        alpha=0.95,
        clip_on=True
    )

# Panel 4 mix drivers (by KEY, so never duplicates)
ax4 = fig.add_subplot(gs[2, :])

labels_wrapped = [wrap_label(n, 26) for n in pivot_names]
cargo_s = pivot.get("cargo_1000_tons", pd.Series(0, index=pivot.index))
teu_s = pivot.get("container_1000_teu", pd.Series(0, index=pivot.index))
pax_s = pivot.get("passengers_count", pd.Series(0, index=pivot.index))

ypos = np.arange(len(labels_wrapped))
ax4.barh(ypos, cargo_s, label="Cargo")
ax4.barh(ypos, teu_s, left=cargo_s, label="Containers")
ax4.barh(ypos, pax_s, left=cargo_s + teu_s, label="Passengers")

ax4.set_yticks(ypos)
ax4.set_yticklabels(labels_wrapped)
ax4.invert_yaxis()
ax4.set_xlabel("Share of mix (%)")
ax4.set_title(f"Mix drivers (Top {topK} by resilience)", loc="left", fontweight="bold", pad=6)
ax4.grid(axis="x")
ax4.grid(axis="y", visible=False)
ax4.legend(ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.05))

out_path = "output/linkedin_seaport_resilience_onepager.png"
fig.savefig(out_path)
plt.close(fig)

print(f"Saved: {out_path}")
print("Saved: output/bundesland_resilience_metrics.csv")
print("Saved: output/bundesland_activity_long.csv")
