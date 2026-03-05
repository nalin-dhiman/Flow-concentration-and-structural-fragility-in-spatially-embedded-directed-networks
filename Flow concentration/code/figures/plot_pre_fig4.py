import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

import matplotlib as mpl
mpl.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3_dir", type=str, required=True)
    parser.add_argument("--defense_dir", type=str, required=True)
    parser.add_argument("--upgrade_dir", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    v3_dir = Path(args.v3_dir)
    defense_dir = Path(args.defense_dir)
    upgrade_dir = Path(args.upgrade_dir)
    assets_dir = Path(__file__).parent / "assets_tables"

    # LOAD DATA
    runs_path = v3_dir / "tables" / "generated_runs_all.csv"
    if not runs_path.exists():
        runs_path = assets_dir / "generative_principle_v3_final_generative_principle_v3_tables_generated_runs_all.csv"
    df_runs = pd.read_csv(runs_path)

    pert_path = defense_dir / "tables" / "natcomm_defense_v2_3_tables_perturbation_results.csv"
    if not pert_path.exists():
        pert_path = assets_dir / "natcomm_defense_v2_3_natcomm_defense_v2_3_tables_perturbation_results.csv"
    df_pert = pd.read_csv(pert_path)

    abl_path = upgrade_dir / "tables" / "ablation_curve_v2.csv"
    if not abl_path.exists():
        abl_path = assets_dir / "ablation_curve_v2.csv"
    if not abl_path.exists():
        try:
            abl_path = list(Path("/home/ub/Downloads/droso_thermo/Phase1_Canonical_Build").rglob("ablation_curve_v2.csv"))[0]
        except IndexError:
            raise FileNotFoundError("Could not locate ablation_curve_v2.csv")
    df_abl = pd.read_csv(abl_path)

    
    fig = plt.figure(figsize=(7.2, 8.0))  
    gs = GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.4, 1.1, 1.1],
        hspace=0.65,   
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c_outer = fig.add_subplot(gs[2, 0])
    ax_c_outer.axis("off")

    
    colors = {
        'targeted_flux':        '#1f77b4', # deep blue (main signal)
        'random':               '#7f7f7f', # medium gray (control)
        'weight_matched_random':'#d95f02', # muted orange (warm accent)
    }
    labels = {
        'targeted_flux':        'Targeted flux',
        'random':               'Random',
        'weight_matched_random':'Weight-matched random',
    }
    
    linestyles = {
        'targeted_flux': '-',
        'random': '--',
        'weight_matched_random': ':'
    }
    zorders = {
        'targeted_flux': 10,
        'random': 8,
        'weight_matched_random': 9
    }

    sn_abl = df_abl[df_abl['condition'].isin(colors.keys())]
    sn_abl = sn_abl[sn_abl['fraction'] <= 0.05]

    for cond in colors:
        subset = sn_abl[sn_abl['condition'] == cond]
        agg = subset.groupby('fraction')['reachability'].mean()
        err = subset.groupby('fraction')['reachability'].std()
        
        ax_a.plot(agg.index, agg.values, label=labels[cond],
                  color=colors[cond], linewidth=2.0, linestyle=linestyles[cond], marker='o', markersize=3.5,
                  zorder=zorders[cond], markerfacecolor='white', markeredgewidth=1.2)
                  
        if not err.isna().all():
            ax_a.fill_between(agg.index, agg.values - err.values,
                              agg.values + err.values, color=colors[cond], alpha=0.15, zorder=zorders[cond]-5)

    ax_a.set_xlim(0, 0.05)
    ax_a.set_ylim(0.012, 0.043)
    ax_a.set_xlabel("Fraction of edges removed $q$")
    ax_a.set_ylabel(r"Target hit probability $R$ ($T_{\max}=50$)")

    ax_a.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),  
        ncol=3, frameon=False,
        handlelength=2.5, columnspacing=1.0, 
        fontsize=8.5
    )
    ax_a.text(0.02, 0.94, "(a)", transform=ax_a.transAxes, ha="left", va="top", fontweight='bold')

   
    axins = inset_axes(
        ax_a, width="42%", height="48%",
        loc="upper right",                      
        bbox_to_anchor=(0.0, 0.0, 0.97, 0.88),  
        bbox_transform=ax_a.transAxes,
        borderpad=0.5,
    )
    for cond in colors:
        subset = sn_abl[sn_abl['condition'] == cond]
        agg = subset.groupby('fraction')['reachability'].mean()
        err = subset.groupby('fraction')['reachability'].std()
        axins.plot(agg.index, agg.values, color=colors[cond],
                   linewidth=1.8, linestyle=linestyles[cond], marker='o', markersize=3.0,
                   zorder=zorders[cond], markerfacecolor='white', markeredgewidth=1.0)
        if not err.isna().all():
            axins.fill_between(agg.index, agg.values - err.values,
                               agg.values + err.values,
                               color=colors[cond], alpha=0.15, zorder=zorders[cond]-5)
    axins.set_xlim(0, 0.01)
    axins.set_ylim(0.012, 0.043)
    axins.tick_params(labelsize=6.5)

   
    def shorten_cond(row):
        pt = row['perturbation_type']
        if pt == 'weight_noise':      return f"wn {row['level']}"
        elif pt == 'edge_drop':       return f"ed {row['level']}"
        elif pt == 'correlated_block':return f"cb {row['level']}"
        return f"{pt} {row['level']}"

    df_pert['Condition'] = df_pert.apply(shorten_cond, axis=1)
    valid_conds = ["wn 0.01","wn 0.05","wn 0.1","ed 0.01","ed 0.05","ed 0.1","cb 0.05"]
    df_pert_sub = df_pert[df_pert['Condition'].isin(valid_conds)].copy()
    df_pert_sub['Condition'] = pd.Categorical(df_pert_sub['Condition'],
                                              categories=valid_conds, ordered=True)
    df_pert_sub = df_pert_sub.sort_values('Condition')

    sns.boxplot(data=df_pert_sub, x="Condition", y="Delta_R",
                ax=ax_b, linewidth=1.2, color="#d9d9d9", linecolor="#4d4d4d", fliersize=0) 
    sns.stripplot(data=df_pert_sub, x="Condition", y="Delta_R",
                  ax=ax_b, color="#000000", alpha=0.25, size=2.5, jitter=True) 
    ax_b.set_xticklabels(ax_b.get_xticklabels(), rotation=35, ha="right")
    ax_b.set_ylabel(r"Relative fragility $\Delta R/R$")
    ax_b.set_xlabel("")
    ax_b.text(0.02, 0.94, "(b)", transform=ax_b.transAxes, ha="left", va="top", fontweight='bold')

    deltas = df_pert_sub['Delta_R']
    ymin = np.percentile(deltas, 1)  - 0.0008
    ymax = np.percentile(deltas, 99) + 0.0008
    ax_b.set_ylim(ymin, ymax)

    
    empirical_val_b = np.median(deltas)
    ax_b.axhline(empirical_val_b, ls="--", color="#1f77b4", lw=1.5, zorder=15, label=r'Empirical $\Delta R/R$') 
    ax_b.legend(loc="lower left", frameon=False, fontsize=8)

    empirical_val_c = 0.63  

    
    vals = df_runs['Relative_Drop'].dropna().values

    
    ax_c1 = inset_axes(
        ax_c_outer, width="100%", height="100%",
        loc="center left",
        bbox_to_anchor=(0.0, 0.0, 0.65, 1.0),   
        bbox_transform=ax_c_outer.transAxes,
        borderpad=0,
    )
    ax_c2 = inset_axes(
        ax_c_outer, width="100%", height="100%",
        loc="center left",
        bbox_to_anchor=(0.69, 0.0, 0.31, 1.0), 
        bbox_transform=ax_c_outer.transAxes,
        borderpad=0,
    )

    ax_c1.hist(vals, bins=22, color="#a6bddb", alpha=0.9, edgecolor="none") 
    ax_c1.set_xlim(0.0, 0.06)
    ax_c1.set_ylabel(f"Null samples (count)\n$n={len(vals)}$")
    ax_c1.set_xlabel(r"Relative fragility $\Delta R/R$")

    ax_c2.set_xlim(0.60, 0.66)
    ax_c2.axvline(empirical_val_c, color="#000000", lw=2.5) 
    ax_c2.set_yticks([])
    ax_c2.set_xticks([0.61, 0.63, 0.65])
    ax_c2.tick_params(axis='x', labelsize=7)
    ax_c2.set_xlabel(r"$\Delta R/R$", fontsize=8)

    ax_c1.spines['right'].set_visible(False)
    ax_c2.spines['left'].set_visible(False)

    d = 0.012
    kw = dict(color='k', clip_on=False, lw=0.9, transform=ax_c1.transAxes)
    ax_c1.plot([1-d, 1+d], [-d,  d ], **kw)
    ax_c1.plot([1-d, 1+d], [1-d, 1+d], **kw)
    kw2 = dict(color='k', clip_on=False, lw=0.9, transform=ax_c2.transAxes)
    ax_c2.plot([-d,  d], [-d,  d ], **kw2)
    ax_c2.plot([-d,  d], [1-d, 1+d], **kw2)

    pct = (np.mean(vals > empirical_val_c) * 100
           if np.mean(vals) < empirical_val_c
           else np.mean(vals < empirical_val_c) * 100)
           
    ax_c2.annotate(f"Empirical\npct: {pct:.1f}%", 
                   xy=(empirical_val_c, 0.85), 
                   xycoords=('data', 'axes fraction'),
                   xytext=(5, 0), textcoords='offset points',
                   ha="left", va="center", fontsize=8, color="#000000", fontweight='bold')

    ax_c1.text(0.02, 0.94, "(c)", transform=ax_c1.transAxes, ha="left", va="top", fontweight='bold')

    

    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.07)

    plt.savefig(args.out_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Figure 4 saved to {args.out_path}")

if __name__ == "__main__":
    main()
