"""QF TSNE in Dyck-matching style (2 clusters, hulls, no axes)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

ROOT = "/n/home07/vrli/hier_gen/question_formation/qf_analysis"
# Use 73-model 300k population (matches 1.png in QF_figures)
phier = np.nan_to_num(np.load(f"{ROOT}/qf_ood_phier_matrix.npy"), nan=0.5)
sm = pd.read_csv(f"{ROOT}/qf_ood_model_summary.csv")

# Use 3-cluster KMeans to match 1.png, then merge Intermediate + Hier as a
# single "Hierarchical" cluster (only 2 hulls / 2 labels in the figure).
km3 = KMeans(n_clusters=3, random_state=0, n_init=20).fit(phier)
df = sm.copy()
df["cluster3"] = km3.labels_
order3 = df.groupby("cluster3")["ood_acc"].mean().sort_values().index.tolist()
relabel3 = {c: rank for rank, c in enumerate(order3)}
df["cluster3_ord"] = df["cluster3"].map(relabel3)
# Merge clusters 1 (intermediate) and 2 (hier) into one "Hierarchical" hull
df["cluster_ord"] = df["cluster3_ord"].apply(lambda c: 0 if c == 0 else 1)
names = {0: "Linear", 1: "Hierarchical"}

coords = TSNE(n_components=2, perplexity=20, random_state=0,
              init="pca", learning_rate="auto").fit_transform(phier)
df["x"] = coords[:, 0]
df["y"] = coords[:, 1]

plt.rcParams["font.family"] = "serif"

fig, ax = plt.subplots(figsize=(9.9, 6.85))

# scatter
sc = ax.scatter(df.x, df.y, c=df.ood_acc, cmap="viridis", vmin=0, vmax=1,
                s=110, linewidths=0)

# hull around each cluster
for c in sorted(df.cluster_ord.unique()):
    sub = df[df.cluster_ord == c][["x", "y"]].values
    if len(sub) < 3:
        continue
    pad_pts = []
    for px, py in sub:
        for dx in (-0.6, 0.6):
            for dy in (-0.6, 0.6):
                pad_pts.append([px + dx, py + dy])
    pad_pts = np.array(pad_pts)
    hull = ConvexHull(pad_pts)
    poly = pad_pts[hull.vertices]
    poly = np.vstack([poly, poly[:1]])
    ax.plot(poly[:, 0], poly[:, 1], color="black", lw=1.0)

# Labels OUTSIDE the hulls. Linear cluster is small in the upper-left; we
# place its label BELOW the hull. Hierarchical cluster is large; we place
# its label above. Both are positioned with a small padding outside the
# convex hull's bounding box.
for c in sorted(df.cluster_ord.unique()):
    sub = df[df.cluster_ord == c]
    cx = sub.x.mean()
    if names[c] == "Linear":
        ly = sub.y.min() - 1.6
        va = "top"
    else:
        ly = sub.y.max() + 1.4
        va = "bottom"
    ax.text(cx, ly, names[c].upper(),
            fontsize=30, ha="center", va=va,
            family="serif", fontweight="bold")

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")
for spine in ax.spines.values():
    spine.set_color("black")
    spine.set_linewidth(0.8)

cbar = plt.colorbar(sc, ax=ax, fraction=0.06, pad=0.02, aspect=18)
cbar.set_label("OOD Accuracy", fontsize=30)
cbar.set_ticks([0.0, 0.5, 1.0])
cbar.ax.tick_params(labelsize=24, width=1.5, length=6)
cbar.outline.set_linewidth(1.2)

plt.tight_layout()
out = "/n/home07/vrli/emnlp_submission/images/qf_tsne_plot.pdf"
plt.savefig(out, bbox_inches='tight')
plt.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
print('Saved', out)
print(df.groupby('cluster_ord')['ood_acc'].agg(['count', 'mean']).round(3))
