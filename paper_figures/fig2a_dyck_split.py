import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams['figure.dpi'] = 300

head_props_df = pd.read_csv("data/transformer_head_properties.csv")


def max_head_score(row, head_type, layer, checkpoint, indist_or_ood):
    if layer < 1 or layer > row['n_layer']:
        return 0
    return max(row[f"cp{checkpoint}_{head_type}_head_l{layer}_h{h}_{indist_or_ood}"] for h in [1, 2, 3, 4])


def has_head_type(row, head_type, checkpoint=5, threshold=0.8, indist_or_ood='indist', layers=(1, 2, 3)):
    return any(max_head_score(row, head_type, l, checkpoint, indist_or_ood) >= threshold for l in layers)


df = head_props_df[head_props_df['n_layer'].isin([2, 3])].copy()

categories = ['Sign-Matching', 'Violation-Detecting', 'Both', 'Neither']
color_map = {
    'Sign-Matching':       '#2072b8',
    'Violation-Detecting': '#cc4d4d',
    'Both':                '0.15',
    'Neither':             '#C4C4C4',
}

plot_data = []
for _, row in df.iterrows():
    # Classify by OOD head presence to match Fig 3 (ablation scatter), which
    # also colors by OOD head type. This gives ~30 violation-detecting models
    # instead of the ~8 we would get with ID classification at the same threshold.
    sign = has_head_type(row, 'sign', threshold=0.8, indist_or_ood='ood')
    neg  = has_head_type(row, 'neg',  threshold=0.8, indist_or_ood='ood')
    if sign and neg:
        label = 'Both'
    elif sign:
        label = 'Sign-Matching'
    elif neg:
        label = 'Violation-Detecting'
    else:
        label = 'Neither'
    plot_data.append({'Head Type': label, 'OOD Acc': row['cp5_ood_acc']})

data = pd.DataFrame(plot_data)
data['Head Type'] = pd.Categorical(data['Head Type'], categories=categories, ordered=True)
data = data.sort_values('Head Type')

fig, ax = plt.subplots(figsize=(7, 5))

sns.boxplot(
    x='OOD Acc',
    y='Head Type',
    data=data,
    orient='h',
    hue='Head Type',
    palette=color_map,
    legend=False,
    linewidth=1.5,
    width=0.5,
    fliersize=0,
    ax=ax,
)
sns.stripplot(
    x='OOD Acc',
    y='Head Type',
    data=data,
    orient='h',
    color='black',
    size=4.0,
    alpha=0.45,
    jitter=0.18,
    ax=ax,
)

for patch in ax.artists:
    patch.set_edgecolor('black')

ax.set_ylabel('Hierarchical Head', fontsize=24)
ax.set_xlabel('OOD Accuracy', fontsize=28)
ax.set_xticks([0, 0.5, 1])
ax.set_xlim(-0.02, 1.02)
ax.tick_params(axis='x', width=1.5, length=5)
ax.tick_params(axis='both', which='major', labelsize=22)

ax.grid(False)
ax.spines['bottom'].set_visible(True)
ax.xaxis.set_ticks_position('bottom')
plt.margins(x=0)

plt.tight_layout()
out_base = "/n/home07/vrli/emnlp_submission/images/dyck_split_by_head_type"
plt.savefig(out_base + ".png", bbox_inches='tight', dpi=300)
plt.savefig(out_base + ".pdf", bbox_inches='tight')
print(f"Saved to {out_base}.png and .pdf")
print(data['Head Type'].value_counts())
