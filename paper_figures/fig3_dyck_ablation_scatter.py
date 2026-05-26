"""Fig 3 scatter (uniform attention ablation) — styled close to the original
transformer_final_scatter_headtype with Violation-Detecting / Sign-Matching
labels and much larger fonts."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams['figure.dpi'] = 300

THRESHOLD = 0.8


def classify(row, indist_or_ood='ood'):
    n_layer = int(row['n_layer'])
    sign = neg = False
    for l in range(1, n_layer + 1):
        for h in range(1, 5):
            sc = f'cp5_sign_head_l{l}_h{h}_{indist_or_ood}'
            nc = f'cp5_neg_head_l{l}_h{h}_{indist_or_ood}'
            if sc in row.index and not pd.isna(row[sc]) and row[sc] >= THRESHOLD:
                sign = True
            if nc in row.index and not pd.isna(row[nc]) and row[nc] >= THRESHOLD:
                neg = True
    if sign and neg:
        return 'Both'
    if sign:
        return 'Sign'
    if neg:
        return 'Viol.'
    return 'Neither'


results = pd.read_csv('03_30_ablation/dyck1/mean_ablation_results.csv')
props = pd.read_csv('data/transformer_head_properties.csv')
merged = results.merge(props, on='id', suffixes=('', '_props'))
merged['head_type_ood'] = merged.apply(lambda r: classify(r, 'ood'), axis=1)

colors = {
    'Sign':    '#2072b8',
    'Viol.':   '#cc4d4d',
    'Both':    '0.10',
    'Neither': '#FFFFFF',
}
zorder = {'Neither': 1, 'Sign': 3, 'Viol.': 3, 'Both': 5}
sizes  = {'Sign': 90, 'Viol.': 90, 'Both': 110, 'Neither': 45}

fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), gridspec_kw={"wspace": 0.30})
for ax_idx, n_layer in enumerate([2, 3]):
    ax = axes[ax_idx]
    sub = merged[merged['n_layer'] == n_layer]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.8)
    for htype in ['Neither', 'Viol.', 'Sign', 'Both']:
        mask = sub['head_type_ood'] == htype
        if mask.sum() == 0:
            continue
        d = sub[mask]
        ax.scatter(d['baseline_ood_acc'], d['mean_ablation_ood_acc'],
                   c=colors[htype], edgecolors='black', linewidths=1.1,
                   s=sizes[htype], zorder=zorder[htype], alpha=0.95)
    ax.set_xlabel('OOD Accuracy Before Ablation', fontsize=26)
    if ax_idx == 0:
        ax.set_ylabel('OOD Accuracy After Ablation', fontsize=26)
    ax.set_title(f'{n_layer}-Layer Models', fontsize=30)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_aspect('equal')
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis='both', labelsize=22, width=1.5, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

# Build legend with every category explicitly, even if absent from a subplot.
legend_order = ['Sign', 'Viol.', 'Both', 'Neither']
handles = [
    Line2D([0], [0], marker='o', linestyle='',
           markerfacecolor=colors[h], markeredgecolor='black',
           markersize={'Sign': 16, 'Viol.': 16, 'Both': 18, 'Neither': 13}[h],
           markeredgewidth=1.2)
    for h in legend_order
]
fig.legend(handles, legend_order, title='Head type', loc='center right',
           bbox_to_anchor=(1.08, 0.5), fontsize=30, title_fontsize=32,
           frameon=True, edgecolor='black', borderpad=0.8,
           labelspacing=1.0, handletextpad=0.7)

plt.subplots_adjust(left=0.07, right=0.84, top=0.92, bottom=0.16, wspace=0.30)
out = '/n/home07/vrli/emnlp_submission/images/dyck_ablation_scatter_headtype.pdf'
plt.savefig(out, bbox_inches='tight')
plt.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
print('Saved', out)
print(merged.groupby(['n_layer', 'head_type_ood']).size())
