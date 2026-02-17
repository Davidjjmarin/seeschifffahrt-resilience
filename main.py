import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import textwrap
import re
import unicodedata

XLSX_PATH = "data/seeschifffahrt.xlsx"

SHEET_CARGO_TONS = "46331-b06"
SHEET_CONTAINER_TEU = "46331-b13"
SHEET_PASSENGERS = "46331-b16"

CURRENT_VALUE_COL = 1
YOY_PCT_COL = 5

Path("output").mkdir(exist_ok=True)


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


def canonical_name(s) -> str:
    s = "" if pd.isna(s) else str(s)
    s = s.replace("\u00a0", " ")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("–", "-").replace("—", "-")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def canonical_key(s) -> str:
    return canonical_name(s).lower()


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

    df = df[~df["bundesland"].str.contains("Ende der Tabelle", case=False, na=False)]
    df = df[~df["bundesland"].str.contains("Insgesamt", case=False, na=False)]
    df = df.dropna(subset=["value"])

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["yoy_pct"] = pd.to_numeric(df["yoy_pct"], errors="coerce")
    df = df.dropna(subset=["value"])

    df["metric"] = metric_name
    return df


economist_style()

cargo = read_bundesland_table(SHEET_CARGO_TONS, "cargo_1000_tons")
teu = read_bundesland_table(SHEET_CONTAINER_TEU, "container_1000_teu")
pax = read_bundesland_table(SHEET_PASSENGERS, "passengers_count")

data_long = pd.concat([cargo, teu, pax], ignore_index=True)

data_long["bundesland_raw"] = data_long["bundesland"]
data_long["bundesland"] = data_long["bundesland"].apply(canonical_name)
data_long["bundesland_key"] = data_long["bundesland"].apply(canonical_key)

# Collapse duplicates even if the visible label looks identical but raw differs
data_long = (
    data_long.groupby(["bundesland_key", "metric"], as_index=False)
             .agg(
                 bundesland=("bundesland", "first"),
                 value=("value", "mean"),
                 yoy_pct=("yoy_pct", "mean")
             )
)

# Optional: remove "Übrige Regionen"
# data_long = data_long[~data_long["bundesland"].str.contains("Übrige Regionen", case=False, na=False)]

data_long.to_csv("output/bundesland_activity_long.csv", index=False)

# Mix proxy + shares
data_long["abs_value"] = data_long["value"].abs()
data_long["total_activity_proxy"] = data_long.groupby("bundesland")["abs_value"].transform("sum")
data_long["share"] = data_long["abs_value"] / (data_long["total_activity_proxy"] + 1e-9)

hhi = (
    data_long.groupby("bundesland")["share"]
    .apply(lambda s: float(np.sum(s ** 2)))
    .reset_index(name="hhi")
)

def top_shares(series: pd.Series) -> pd.Series:
    s = series.sort_values(ascending=False).reset_index(drop=True)
    top1 = float(s.iloc[0]) if len(s) else np.nan
    top2 = float(s.iloc[:2].sum()) if len(s) >= 2 else float(s.sum())
    return pd.Series({"top1_share": top1, "top2_share": top2})

tops = (
    data_long.groupby("bundesland")["share"]
    .apply(top_shares)
    .reset_index()
)

momentum = (
    data_long.groupby("bundesland")["yoy_pct"]
    .mean()
    .reset_index(name="avg_yoy_pct")
)

tot_activity = (
    data_long.groupby("bundesland")["abs_value"]
    .sum()
    .reset_index(name="total_activity_proxy")
)

metrics = (
    hhi.merge(tops, on="bundesland")
       .merge(momentum, on="bundesland", how="left")
       .merge(tot_activity, on="bundesland", how="left")
)

metrics["avg_yoy_pct"] = metrics["avg_yoy_pct"].fillna(0.0)
metrics["diversification"] = 1 - metrics["hhi"]

minv, maxv = metrics["diversification"].min(), metrics["diversification"].max()
metrics["diversification_score"] = 100 * (metrics["diversification"] - minv) / (maxv - minv + 1e-9)

m = np.clip(metrics["avg_yoy_pct"], -20, 20)
metrics["momentum_score"] = (m + 20) * (100 / 40)

metrics["resilience_score"] = 0.85 * metrics["diversification_score"] + 0.15 * metrics["momentum_score"]
metrics = metrics.sort_values("resilience_score", ascending=False)

metrics.to_csv("output/bundesland_resilience_metrics.csv", index=False)

# Rankings
N = min(10, len(metrics))
top_res = metrics.head(N).sort_values("resilience_score", ascending=True)
top_dep = metrics.sort_values("hhi", ascending=False).head(N).sort_values("hhi", ascending=True)

# Mix pivot
mix = data_long.copy()
mix["share_pct"] = 100 * mix["share"]

pivot = (
    mix.pivot_table(index="bundesland", columns="metric", values="share_pct", aggfunc="sum")
       .fillna(0)
)
pivot = pivot.groupby(pivot.index).sum()

topK = min(8, len(metrics))
top_names = metrics.head(topK)["bundesland"].tolist()
pivot = pivot.loc[[n for n in top_names if n in pivot.index]]

# Figure
fig = plt.figure(figsize=(12.5, 15), dpi=240)
fig.subplots_adjust(left=0.20, right=0.97, top=0.90, bottom=0.06, hspace=0.55, wspace=0.35)
gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.55, 1.25])

fig.text(0.20, 0.965, "German seaport regions: resilience vs concentration", ha="left", va="top",
         fontsize=18, fontweight="bold")
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

# Panel 3 scatter + ALL labels
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

# Panel 4 mix drivers
ax4 = fig.add_subplot(gs[2, :])

labels = [wrap_label(x, 26) for x in pivot.index.tolist()]
cargo_s = pivot.get("cargo_1000_tons", pd.Series(0, index=pivot.index))
teu_s = pivot.get("container_1000_teu", pd.Series(0, index=pivot.index))
pax_s = pivot.get("passengers_count", pd.Series(0, index=pivot.index))

ypos = np.arange(len(labels))
ax4.barh(ypos, cargo_s, label="Cargo")
ax4.barh(ypos, teu_s, left=cargo_s, label="Containers")
ax4.barh(ypos, pax_s, left=cargo_s + teu_s, label="Passengers")

ax4.set_yticks(ypos)
ax4.set_yticklabels(labels)
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
