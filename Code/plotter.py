"""Plot results from paper_experiments.py.

Usage:
    python plotter.py --experiment {1,2,3} [--input_dir results] [--output_dir figures]
    python plotter.py --experiment all [--input_dir results] [--output_dir figures]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Consistent style across all plots
PLACEMENT_STYLES = {
    'offline':      {'color': '#1f77b4', 'marker': 'o', 'label': 'Offline'},
    'fluid':        {'color': '#ff7f0e', 'marker': 's', 'label': 'Fluid'},
    'myopic':       {'color': '#d62728', 'marker': 'D', 'label': 'Myopic Pl.'},
}

PLACEMENT_ORDER = ['offline', 'fluid', 'myopic']


def load_csv(input_dir: str, experiment: int) -> pd.DataFrame:
    path = os.path.join(input_dir, f'experiment_{experiment}.csv')
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Experiment 1: d-sensitivity
# ---------------------------------------------------------------------------

def plot_experiment_1(input_dir: str, output_dir: str):
    """Plot competitive ratio vs d, one figure per demand model."""
    df = load_csv(input_dir, 1)
    os.makedirs(output_dir, exist_ok=True)

    demand_models = sorted(df['demand_model'].unique())
    fulfillments = ['Myopic', 'O-SP']
    load_factors = sorted(df['load_factor'].unique())
    d_values = sorted(df['d'].unique())

    for dm in demand_models:
        df_dm = df[df['demand_model'] == dm]

        # --- Main grid: rows=fulfillment, cols=load factor ---
        fig, axes = plt.subplots(
            len(fulfillments), len(load_factors),
            figsize=(3.2 * len(load_factors), 3.0 * len(fulfillments)),
            sharex=True, sharey=True,
        )
        if len(fulfillments) == 1:
            axes = axes[np.newaxis, :]

        for row, ful in enumerate(fulfillments):
            for col, lf in enumerate(load_factors):
                ax = axes[row, col]
                subset = df_dm[(df_dm['fulfillment'] == ful) & (df_dm['load_factor'] == lf)]

                for placement in PLACEMENT_ORDER:
                    ps = subset[subset['placement'] == placement]
                    means = ps.groupby('d')['competitive_ratio'].mean()
                    sems = ps.groupby('d')['competitive_ratio'].sem()
                    style = PLACEMENT_STYLES[placement]
                    ax.errorbar(
                        means.index, means.values, yerr=sems.values,
                        color=style['color'], marker=style['marker'],
                        label=style['label'], capsize=3, markersize=5, linewidth=1.2,
                    )

                if row == 0:
                    ax.set_title(f'Load = {lf}', fontsize=10)
                if row == len(fulfillments) - 1:
                    ax.set_xlabel('d (degree)')
                    ax.set_xticks(d_values)
                if col == 0:
                    ax.set_ylabel(f'{ful} fulfillment\nCompetitive ratio')
                ax.grid(True, alpha=0.3)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=len(PLACEMENT_ORDER),
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f'{dm}', fontsize=12, y=1.05)
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        path = os.path.join(output_dir, f'experiment_1_d_sensitivity_{dm}.pdf')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {path}')

        # --- Companion: fulfillment gap (O-SP - Myopic) ---
        fig2, axes2 = plt.subplots(
            1, len(load_factors),
            figsize=(3.2 * len(load_factors), 3.0),
            sharex=True, sharey=True,
        )
        if len(load_factors) == 1:
            axes2 = [axes2]

        for col, lf in enumerate(load_factors):
            ax = axes2[col]
            sub_lf = df_dm[df_dm['load_factor'] == lf]

            for placement in PLACEMENT_ORDER:
                ps = sub_lf[sub_lf['placement'] == placement]
                pivot = ps.pivot_table(
                    index=['d', 'instance_id'], columns='fulfillment',
                    values='competitive_ratio'
                )
                pivot['gap'] = pivot['O-SP'] - pivot['Myopic']
                gap_stats = pivot.groupby('d')['gap']
                means = gap_stats.mean()
                sems = gap_stats.sem()
                style = PLACEMENT_STYLES[placement]
                ax.errorbar(
                    means.index, means.values, yerr=sems.values,
                    color=style['color'], marker=style['marker'],
                    label=style['label'], capsize=3, markersize=5, linewidth=1.2,
                )

            ax.set_title(f'Load = {lf}', fontsize=10)
            ax.set_xlabel('d (degree)')
            ax.set_xticks(d_values)
            if col == 0:
                ax.set_ylabel('O-SP − Myopic gap')
            ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
            ax.grid(True, alpha=0.3)

        handles, labels = axes2[0].get_legend_handles_labels()
        fig2.legend(handles, labels, loc='upper center', ncol=len(PLACEMENT_ORDER),
                    fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.05))
        fig2.suptitle(f'{dm}', fontsize=12, y=1.10)
        fig2.tight_layout(rect=[0, 0, 1, 0.92])

        path2 = os.path.join(output_dir, f'experiment_1_fulfillment_gap_{dm}.pdf')
        fig2.savefig(path2, bbox_inches='tight')
        plt.close(fig2)
        print(f'Saved {path2}')


# ---------------------------------------------------------------------------
# Experiment 2: Sample convergence
# ---------------------------------------------------------------------------

def plot_experiment_2(input_dir: str, output_dir: str):
    """Plot competitive ratio vs K, one figure per demand model."""
    df = load_csv(input_dir, 2)
    os.makedirs(output_dir, exist_ok=True)

    demand_models = sorted(df['demand_model'].unique())
    weight_settings = sorted(df['weight_setting'].unique())
    load_factors = sorted(df['load_factor'].unique())
    fulfillments = ['Myopic', 'O-SP']
    K_values = sorted(df['K'].unique())

    n_cols = len(load_factors) * len(weight_settings)
    n_rows = len(fulfillments)

    for dm in demand_models:
        df_dm = df[df['demand_model'] == dm]

        # --- Main grid ---
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.5 * n_cols, 3.0 * n_rows),
            sharex=True, sharey=True,
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for row, ful in enumerate(fulfillments):
            col_idx = 0
            for ws in weight_settings:
                for lf in load_factors:
                    ax = axes[row, col_idx]
                    subset = df_dm[
                        (df_dm['fulfillment'] == ful) &
                        (df_dm['weight_setting'] == ws) &
                        (df_dm['load_factor'] == lf)
                    ]

                    for placement in PLACEMENT_ORDER:
                        ps = subset[subset['placement'] == placement]
                        means = ps.groupby('K')['competitive_ratio'].mean()
                        sems = ps.groupby('K')['competitive_ratio'].sem()
                        style = PLACEMENT_STYLES[placement]
                        ax.errorbar(
                            means.index, means.values, yerr=sems.values,
                            color=style['color'], marker=style['marker'],
                            label=style['label'], capsize=3, markersize=5, linewidth=1.2,
                        )

                    if row == 0:
                        ax.set_title(f'{ws.title()}, load={lf}', fontsize=10)
                    if row == n_rows - 1:
                        ax.set_xlabel('K (samples)')
                        ax.set_xscale('log')
                        ax.set_xticks(K_values)
                        ax.set_xticklabels([str(k) for k in K_values])
                    if col_idx == 0:
                        ax.set_ylabel(f'{ful} fulfillment\nCompetitive ratio')
                    ax.grid(True, alpha=0.3)

                    col_idx += 1

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=len(PLACEMENT_ORDER),
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f'{dm}', fontsize=12, y=1.05)
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        path = os.path.join(output_dir, f'experiment_2_sample_convergence_{dm}.pdf')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {path}')

        # --- Companion: Offline - Fluid gap vs K ---
        fig2, axes2 = plt.subplots(
            n_rows, n_cols,
            figsize=(3.5 * n_cols, 3.0 * n_rows),
            sharex=True, sharey=True,
        )
        if n_rows == 1:
            axes2 = axes2[np.newaxis, :]

        for row, ful in enumerate(fulfillments):
            col_idx = 0
            for ws in weight_settings:
                for lf in load_factors:
                    ax = axes2[row, col_idx]
                    subset = df_dm[
                        (df_dm['fulfillment'] == ful) &
                        (df_dm['weight_setting'] == ws) &
                        (df_dm['load_factor'] == lf)
                    ]

                    off = subset[subset['placement'] == 'offline'].set_index(
                        ['K', 'instance_id'])['competitive_ratio']
                    flu = subset[subset['placement'] == 'fluid'].set_index(
                        ['K', 'instance_id'])['competitive_ratio']
                    gap = (off - flu).reset_index()
                    gap.columns = ['K', 'instance_id', 'gap']

                    means = gap.groupby('K')['gap'].mean()
                    sems = gap.groupby('K')['gap'].sem()
                    ax.errorbar(
                        means.index, means.values, yerr=sems.values,
                        color='#1f77b4', marker='o', capsize=3, markersize=5, linewidth=1.2,
                    )
                    ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')

                    if row == 0:
                        ax.set_title(f'{ws.title()}, load={lf}', fontsize=10)
                    if row == n_rows - 1:
                        ax.set_xlabel('K (samples)')
                        ax.set_xscale('log')
                        ax.set_xticks(K_values)
                        ax.set_xticklabels([str(k) for k in K_values])
                    if col_idx == 0:
                        ax.set_ylabel(f'{ful} fulfillment\nOffline − Fluid gap')
                    ax.grid(True, alpha=0.3)

                    col_idx += 1

        fig2.suptitle(f'{dm}', fontsize=12, y=1.02)
        fig2.tight_layout(rect=[0, 0, 1, 0.96])
        path2 = os.path.join(output_dir, f'experiment_2_offline_fluid_gap_{dm}.pdf')
        fig2.savefig(path2, bbox_inches='tight')
        plt.close(fig2)
        print(f'Saved {path2}')


# ---------------------------------------------------------------------------
# Experiment 3: Placement x Fulfillment interaction
# ---------------------------------------------------------------------------

def plot_experiment_3(input_dir: str, output_dir: str):
    """Plot grouped bar charts, one per demand model."""
    df = load_csv(input_dir, 3)
    os.makedirs(output_dir, exist_ok=True)

    demand_models = sorted(df['demand_model'].unique())
    fulfillments = ['Myopic', 'O-SP']
    x = np.arange(len(PLACEMENT_ORDER))
    bar_width = 0.35

    for dm in demand_models:
        df_dm = df[df['demand_model'] == dm]

        # --- Main: grouped bar chart ---
        agg = df_dm.groupby(['placement', 'fulfillment'])['competitive_ratio'].agg(['mean', 'sem'])
        agg = agg.reset_index()

        fig, ax = plt.subplots(figsize=(7, 4))

        for i, ful in enumerate(fulfillments):
            sub = agg[agg['fulfillment'] == ful]
            sub = sub.set_index('placement').loc[PLACEMENT_ORDER]
            ax.bar(
                x + i * bar_width, sub['mean'], bar_width,
                yerr=sub['sem'], capsize=4,
                label=ful,
                color=['#4a90d9', '#e8843c'][i],
                edgecolor='white', linewidth=0.5,
            )

        ax.set_xlabel('Placement procedure')
        ax.set_ylabel('Competitive ratio')
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels([PLACEMENT_STYLES[p]['label'] for p in PLACEMENT_ORDER])
        ax.set_title(f'{dm}', fontsize=12)
        ax.legend(title='Fulfillment')
        ax.grid(True, axis='y', alpha=0.3)

        fig.tight_layout()
        path = os.path.join(output_dir, f'experiment_3_placement_fulfillment_{dm}.pdf')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {path}')

        # --- Companion: fulfillment uplift ---
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))

        pivot = df_dm.pivot_table(
            index=['placement', 'instance_id'], columns='fulfillment',
            values='competitive_ratio'
        )
        pivot['uplift'] = pivot['O-SP'] - pivot['Myopic']
        uplift = pivot.groupby('placement')['uplift'].agg(['mean', 'sem'])
        uplift = uplift.loc[PLACEMENT_ORDER]

        colors = [PLACEMENT_STYLES[p]['color'] for p in PLACEMENT_ORDER]
        labels = [PLACEMENT_STYLES[p]['label'] for p in PLACEMENT_ORDER]
        ax2.bar(
            x, uplift['mean'], 0.5, yerr=uplift['sem'], capsize=4,
            color=colors, edgecolor='white', linewidth=0.5,
        )
        ax2.set_xlabel('Placement procedure')
        ax2.set_ylabel('O-SP − Myopic uplift')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_title(f'{dm}', fontsize=12)
        ax2.axhline(0, color='gray', linewidth=0.8, linestyle=':')
        ax2.grid(True, axis='y', alpha=0.3)

        fig2.tight_layout()
        path2 = os.path.join(output_dir, f'experiment_3_fulfillment_uplift_{dm}.pdf')
        fig2.savefig(path2, bbox_inches='tight')
        plt.close(fig2)
        print(f'Saved {path2}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plot paper experiment results')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['1', '2', '3', 'all'],
                        help='Which experiment to plot')
    parser.add_argument('--input_dir', type=str, default='results',
                        help='Directory with experiment CSV files')
    parser.add_argument('--output_dir', type=str, default='figures',
                        help='Directory for output figures')
    args = parser.parse_args()

    experiments = ['1', '2', '3'] if args.experiment == 'all' else [args.experiment]

    plotters = {
        '1': plot_experiment_1,
        '2': plot_experiment_2,
        '3': plot_experiment_3,
    }

    for exp in experiments:
        print(f'Plotting experiment {exp}...')
        plotters[exp](args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
