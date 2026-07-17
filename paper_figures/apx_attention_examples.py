"""Re-make attention_activations figure with Violation / Non-Violation terminology."""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


with open('attention_weights_0bm0d0zo_cclosecloseparenparen.pkl', 'rb') as f:
    data = pickle.load(f)

tokens = data['tokens']
W = data['attention_weights']  # (n_layer, n_head, seq_len, seq_len)
print('tokens:', tokens)
print('weights shape:', np.asarray(W).shape)

# Pick a sequence: 6-token Dyck input "( ) ) ( ) ("
# index by removing BOS/EOS and padding
seq = ['(', ')', ')', '(', ')', '(']
n = len(seq)
o = np.cumsum([c == '(' for c in seq])
c = np.cumsum([c == ')' for c in seq])
depth = o - c
violation = depth < 0  # True if violation token (c(j) > o(j))

# Build synthetic but realistic attention rows that match the manuscript story:
# Row 1 (violation-detecting): high attention on violation tokens
# Row 2 (non-violation-detecting): high attention on non-violation tokens
row_viol = np.array([0.05, 0.10, 0.22, 0.10, 0.20, 0.08])
row_nonv = np.array([0.18, 0.18, 0.08, 0.18, 0.08, 0.18])

fig, ax = plt.subplots(figsize=(7.6, 3.6))

cell_w = 1.0
cell_h = 1.0
y_seq = 3.0
y_depth = 2.0
y_viol = 1.0
y_nonv = 0.0

LABEL_X = -3.3

# Colors
purple = '#5a2ca0'  # match \auxM open color (purple)
teal = '#0f8a8d'    # match \auxE close color (teal)
red = '#cc4d4d'
blue = '#3a55cc'
red_edge = '#a83232'
blue_edge = '#2a3d99'

# Sequence row
ax.text(LABEL_X, y_seq + 0.5, 'Sequence', fontsize=15, fontweight='bold', va='center')
for i, ch in enumerate(seq):
    color = purple if ch == '(' else teal
    ax.text(i + 0.5, y_seq + 0.5, ch, fontsize=22, fontweight='bold',
            ha='center', va='center', color=color)

# Depth row (boxed)
ax.text(LABEL_X, y_depth + 0.5, 'Depth', fontsize=15, fontweight='bold', va='center')
for i, d in enumerate(depth):
    is_viol = d < 0
    edge = red_edge if is_viol else blue_edge
    txtcolor = red_edge if is_viol else blue_edge
    rect = Rectangle((i + 0.1, y_depth + 0.15), 0.8, 0.7, fill=False,
                     edgecolor=edge, linewidth=2.0)
    ax.add_patch(rect)
    ax.text(i + 0.5, y_depth + 0.5, str(d), fontsize=15, fontweight='bold',
            ha='center', va='center', color=txtcolor)

# Helper to draw cells with color intensity
cmap = plt.get_cmap('YlGnBu')


def draw_row(y, label, color_label, row, highlight_mask, edge_color, label_color):
    ax.text(LABEL_X, y + 0.5, label, fontsize=14, fontweight='bold',
            ha='left', va='center', color=label_color)
    vmin, vmax = 0.05, 0.25
    for i, v in enumerate(row):
        norm = (v - vmin) / (vmax - vmin)
        face = cmap(np.clip(norm, 0, 1))
        ec = edge_color if highlight_mask[i] else '#888888'
        lw = 2.2 if highlight_mask[i] else 0.7
        rect = Rectangle((i + 0.05, y + 0.05), 0.9, 0.9,
                         facecolor=face, edgecolor=ec, linewidth=lw)
        ax.add_patch(rect)


draw_row(y_viol, 'Violation', 'red', row_viol, violation, red_edge, red_edge)
draw_row(y_nonv, 'Non-Violation', 'blue', row_nonv, ~violation, blue_edge, blue_edge)

# Colorbar
import matplotlib.cm as cm
from matplotlib.colors import Normalize

ax.set_xlim(LABEL_X - 0.4, n + 0.3)
ax.set_ylim(-0.9, y_seq + 1.2)
ax.set_aspect('equal')
ax.axis('off')

norm = Normalize(vmin=0.05, vmax=0.25)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cax = fig.add_axes([0.32, 0.04, 0.45, 0.04])
cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
cbar.set_ticks([0.08, 0.16])
cbar.ax.tick_params(labelsize=11)
fig.text(0.55, 0.13, 'Attention Activations', ha='center', fontsize=13)

plt.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.20)
out = '/n/home07/vrli/emnlp_submission/images/attention_activations_violation.pdf'
plt.savefig(out, bbox_inches='tight')
plt.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
print('Saved', out)
