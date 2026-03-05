#!/usr/bin/env python3
"""
fig5_v5.py  –  definitive version with clean external legend strips.

Layout (3-row GridSpec):
  Row 0  →  Panel (a) full-width ECDF         legend floats above via transAxes
  Row 1  →  Panel (b) left | Panel (c) right
  Row 2  →  Legend strip (b) | Legend strip (c)   [dedicated axes, axis off]

All legends are OUTSIDE the data panels and cannot overlap any curve or bar.
A thin ·····separator line is drawn between row-1 axes and row-2 strip.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from pathlib import Path
import matplotlib as mpl

# ── rcParams ──────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":          "DejaVu Sans",
    "font.size":            9,
    "axes.labelsize":       9,
    "axes.titlesize":       9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      7.5,
    "legend.handlelength":  1.8,
    "axes.linewidth":       0.8,
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
})

# ── Colour palette ─────────────────────────────────────────────────────────────
C_BASE   = '#F4A261'   # warm orange  – Base / N2 ECDF
C_GEN01  = '#2A9D8F'   # teal         – Gen f=0.01
C_GEN05  = '#E76F51'   # coral-red    – Gen f=0.05
C_REAL   = '#264653'   # dark slate   – Empirical (thickest)

MB_COLS  = ['#2A9D8F', '#E76F51', '#F4A261', '#9B5DE5']   # panel-b line colours
MB_LS    = ['-',       '--',      ':',       '-.']         # panel-b linestyles

C_EMP_B  = '#264653'   # bar: Empirical
C_N2_B   = '#74C2C4'   # bar: N2
C_GEN_B  = '#F4A261'   # bar: Generated


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_csv(*candidates):
    for p in candidates:
        p = Path(p)
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"None found: {candidates}")


def ecdf_xy(series):
    v = np.sort(series.dropna().values)
    return v, np.arange(1, len(v) + 1) / len(v)


def safe_norm(col, base_val, real_val):
    d = real_val - base_val
    if pd.notna(d) and abs(d) > 1e-4:
        return (col - base_val) / d
    if pd.notna(real_val) and abs(real_val) > 1e-9:
        return col / real_val
    return col


def topo_mean(df, col, pat):
    m = df['Model'].str.contains(pat, case=False, na=False, regex=True)
    s = df.loc[m, col]
    return float(s.mean()) if not s.empty else np.nan


# ═════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3_dir",   required=True)
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    v3   = Path(args.v3_dir)
    adir = Path(__file__).parent / "assets_tables"

    ecdf_df = load_csv(adir / "fig5_ecdf.csv")
    real_df = load_csv(v3   / "tables/real_metrics_aligned.csv")
    n2_df   = load_csv(v3   / "tables/n2_reference.csv")
    gen_df  = load_csv(v3   / "tables/generated_runs_all.csv")
    topo_df = load_csv(v3   / "tables/topology_summary_by_f.csv")
    gen_df  = gen_df.reset_index(drop=True)

    # ── Figure & GridSpec ─────────────────────────────────────────────────────
    # 3 rows: [panel-a] / [panel-b  panel-c] / [legend-b  legend-c]
    fig = plt.figure(figsize=(7.2, 8.2))
    gs  = mgridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.4, 1.05, 0.35],   # legend strip = 0.35 rel units ≈ 0.62 in
        hspace=0.52,
        wspace=0.40,
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.03)

    ax_a  = fig.add_subplot(gs[0, :])    # full-width top
    ax_b  = fig.add_subplot(gs[1, 0])
    ax_c  = fig.add_subplot(gs[1, 1])
    ax_lb = fig.add_subplot(gs[2, 0])    # legend host – panel b
    ax_lc = fig.add_subplot(gs[2, 1])    # legend host – panel c
    ax_lb.axis('off')
    ax_lc.axis('off')

    # ══════════════════════════════════════════════════════════════════════════
    # Panel (a) – ECDF of edge flux
    # ══════════════════════════════════════════════════════════════════════════
    spec_a = [
        ('Base',     C_BASE,  '--',  1.5, 'Base ($N_2$)'),
        ('Gen_0.01', C_GEN01, '-.',  1.6, r'Gen ($f{=}0.01$)'),
        ('Gen_0.05', C_GEN05, ':',   1.8, r'Gen ($f{=}0.05$)'),
        ('REAL',     C_REAL,  '-',   2.4, 'Empirical connectome'),
    ]
    leg_a = []
    for col, color, ls, lw, lbl in spec_a:
        if col not in ecdf_df.columns:
            continue
        x, y = ecdf_xy(ecdf_df[col])
        ax_a.plot(x, y, color=color, ls=ls, lw=lw,
                  zorder=5 if col == 'REAL' else 3)
        leg_a.append(Line2D([0], [0], color=color, ls=ls, lw=lw, label=lbl))

    ax_a.set_xscale('log')
    ax_a.set_ylim(-0.03, 1.05)
    ax_a.set_xlabel("Target-conditioned flux $J_{ij}$")
    ax_a.set_ylabel("Proportion")

    # Legend centred above panel (a) – in the generous top margin (top=0.92)
    ax_a.legend(
        handles=leg_a,
        loc='lower center',
        bbox_to_anchor=(0.50, 1.02),
        bbox_transform=ax_a.transAxes,
        ncol=4, frameon=False,
        handlelength=2.0, columnspacing=1.0, fontsize=8,
    )
    # Panel label: to the left of and above the axes
    ax_a.text(-0.08, 1.12, "(a)", transform=ax_a.transAxes,
              ha='left', va='top', fontsize=11, fontweight='bold')

    # ══════════════════════════════════════════════════════════════════════════
    # Panel (b) – normalised metric shift vs rewiring fraction f
    # ══════════════════════════════════════════════════════════════════════════
    metrics_b = {
        'DNP11_E_weight':  'DNP11 enrichment',
        'flux_share_top1': 'Top-1% flux share',
        'Base_R50':        'Reachability',
        'Relative_Drop':   'Fragility',
    }
    leg_b_handles = []
    for i, (col, lbl) in enumerate(metrics_b.items()):
        if col not in gen_df.columns:
            continue
        rv = float(real_df[col].iloc[0]) if col in real_df.columns else np.nan
        bv = float(n2_df[col].iloc[0])   if col in n2_df.columns  else np.nan
        yn = safe_norm(gen_df[col], bv, rv)

        grp = pd.DataFrame({'y': yn, 'f': gen_df['f_val']}).groupby('f')
        agg = grp['y'].mean().clip(-0.6, 2.2)
        err = grp['y'].std().fillna(0)
        lo  = (agg - err).clip(-0.6, 2.2)
        hi  = (agg + err).clip(-0.6, 2.2)

        c, ls = MB_COLS[i % 4], MB_LS[i % 4]
        ax_b.plot(agg.index, agg.values, color=c, lw=1.7, ls=ls)
        ax_b.fill_between(agg.index, lo, hi, color=c, alpha=0.13)
        leg_b_handles.append(Line2D([0], [0], color=c, lw=1.7, ls=ls, label=lbl))

    ax_b.axhline(1.0, color='#444444', ls='--', lw=0.9)
    ax_b.axhline(0.0, color='#AAAAAA', ls=':',  lw=0.9)
    leg_b_handles += [
        Line2D([0], [0], color='#444444', ls='--', lw=0.9, label='Empirical ref.'),
        Line2D([0], [0], color='#AAAAAA', ls=':',  lw=0.9, label='$N_2$ baseline'),
    ]

    ax_b.set_ylim(-0.6, 2.2)
    ax_b.set_xlabel(r"Rewiring fraction $f$")
    ax_b.set_ylabel("Normalised metric shift")
    ax_b.yaxis.set_label_coords(-0.20, 0.5)
    ax_b.text(-0.20, 1.07, "(b)", transform=ax_b.transAxes,
              ha='left', va='top', fontsize=11, fontweight='bold')

    # ══════════════════════════════════════════════════════════════════════════
    # Panel (c) – topology bar chart
    # ══════════════════════════════════════════════════════════════════════════
    topo_cols = {
        'deg_mean':          'Out-deg.',
        'modularity_proxy':  'Modularity',
        'reciprocity':       'Reciprocity',
        'rich_club_1':       'Rich-club',
    }
    model_order = ['Empirical', '$N_2$', 'Generated']
    bar_pal     = {'Empirical': C_EMP_B, '$N_2$': C_N2_B, 'Generated': C_GEN_B}

    bar_rows = []
    for col, nice in topo_cols.items():
        rv = (float(real_df[col].iloc[0]) if col in real_df.columns
              else topo_mean(topo_df, col, r'REAL|Empirical'))
        bv = (float(n2_df[col].iloc[0])   if col in n2_df.columns
              else topo_mean(topo_df, col, r'N2|Base'))
        gv = (float(gen_df[col].median()) if col in gen_df.columns
              else topo_mean(topo_df, col, r'Gen|generated'))
        if not rv or not np.isfinite(rv):
            continue
        bar_rows.extend([
            {'Metric': nice, 'Model': 'Empirical', 'Value': 1.0},
            {'Metric': nice, 'Model': '$N_2$',      'Value': bv/rv if pd.notna(bv) else np.nan},
            {'Metric': nice, 'Model': 'Generated',  'Value': gv/rv if pd.notna(gv) else np.nan},
        ])

    bdf = pd.DataFrame(bar_rows)
    if not bdf.empty:
        metrics_order = list(dict.fromkeys(bdf['Metric']))
        n_mo  = len(model_order)
        width = 0.23
        x     = np.arange(len(metrics_order))

        pivot = (bdf.pivot_table(index='Metric', columns='Model',
                                 values='Value', aggfunc='first')
                    .reindex(index=metrics_order, columns=model_order))

        for j, model in enumerate(model_order):
            offset = (j - (n_mo - 1) / 2) * width
            vals   = np.nan_to_num(pivot[model].values.astype(float))
            ax_c.bar(x + offset, vals, width=width,
                     color=bar_pal[model], edgecolor='white', linewidth=0.5)

        ax_c.axhline(1.0, color='#444444', lw=0.9, ls='--', zorder=0)
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(metrics_order, rotation=22, ha='right', fontsize=8)
        ax_c.set_ylabel("Value relative to empirical")
        ax_c.set_ylim(0, 1.22)
    else:
        ax_c.text(0.5, 0.5, "Topology data\nunavailable",
                  transform=ax_c.transAxes, ha='center', va='center',
                  color='gray', fontsize=9)

    ax_c.text(-0.20, 1.07, "(c)", transform=ax_c.transAxes,
              ha='left', va='top', fontsize=11, fontweight='bold')

    # ══════════════════════════════════════════════════════════════════════════
    # Legend strips (row 2) – clean, outside all data axes
    # ══════════════════════════════════════════════════════════════════════════

    # Draw a thin horizontal separator line across full figure width
    # between row-1 axes bottom and row-2 strip top
    fig.canvas.draw()
    lb_pos = ax_lb.get_position()
    ab_pos = ax_b.get_position()
    sep_y  = (lb_pos.y1 + ab_pos.y0) / 2.0     # midpoint in figure fraction
    sep_line = plt.Line2D(
        [0.07, 0.97], [sep_y, sep_y],
        transform=fig.transFigure,
        color='#cccccc', lw=0.8, clip_on=False,
    )
    fig.add_artist(sep_line)

    # Small italic sub-heading labels
    ax_lb.text(0.01, 1.0, "Panel (b):", transform=ax_lb.transAxes,
               ha='left', va='top', fontsize=6.5, color='#888888', style='italic')
    ax_lc.text(0.01, 1.0, "Panel (c):", transform=ax_lc.transAxes,
               ha='left', va='top', fontsize=6.5, color='#888888', style='italic')

    # Panel-b legend: 3 columns, no frame
    ax_lb.legend(
        handles=leg_b_handles,
        loc='center',
        bbox_to_anchor=(0.50, 0.45),
        bbox_transform=ax_lb.transAxes,
        ncol=3, frameon=False,
        fontsize=7.5, handlelength=1.8,
        labelspacing=0.40, columnspacing=1.1,
    )

    # Panel-c legend: 3 columns, no frame
    leg_c = [mpatches.Patch(color=bar_pal[m], label=m) for m in model_order]
    ax_lc.legend(
        handles=leg_c,
        loc='center',
        bbox_to_anchor=(0.50, 0.45),
        bbox_transform=ax_lc.transAxes,
        ncol=3, frameon=False,
        fontsize=7.5, handlelength=1.2,
        labelspacing=0.40, columnspacing=1.1,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.savefig(args.out_path, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved → {args.out_path}")


if __name__ == "__main__":
    main()