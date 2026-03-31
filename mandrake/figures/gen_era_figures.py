import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        plt.style.use('default')

OUT = Path(__file__).parent

# ── FIGURE 1: Era 1 (v1–v7) ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle(
    "Era 1 Development: Optimising for Macro-F1 and Retroviral Coverage (v1–v7)",
    fontsize=13, fontweight='bold', y=0.98)
fig.text(0.5, 0.935,
    "Final submission: MF1=0.650  |  Retroviral AUC=0.833  |  10/12 Retroviral TPs",
    ha='center', fontsize=10, color='#444444')

stages     = ['v1', 'v3', 'v4', 'v5', 'v6/v7']
x          = np.arange(len(stages))
mf1        = [0.634, 0.634, 0.607, 0.676, 0.650]
retro_auc_x = [0, 4]
retro_auc_y = [0.627, 0.833]

ax1.axvspan(3.55, 4.45, alpha=0.12, color='#2ca02c', zorder=0)
ax1.axvline(x=3.5, color='grey', linestyle='--', linewidth=1.2, zorder=1)
ax1.text(3.55, 0.564, "Rank averaging\nintroduced", fontsize=8,
         color='grey', rotation=90, va='bottom')

ax1.plot(x, mf1, 'b-o', linewidth=2, markersize=7, label='Macro-F1', zorder=3)
ax1.set_ylim(0.55, 0.73)
ax1.set_ylabel('Macro-F1', color='#1f77b4', fontsize=11)
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.set_xticks(x)
ax1.set_xticklabels(stages, fontsize=11)

ax1r = ax1.twinx()
ax1r.plot(retro_auc_x, retro_auc_y, 's--', color='#ff7f0e',
          linewidth=2, markersize=7, label='Retroviral AUC', zorder=3)
ax1r.set_ylim(0.50, 0.95)
ax1r.set_ylabel('Retroviral AUC', color='#ff7f0e', fontsize=11)
ax1r.tick_params(axis='y', labelcolor='#ff7f0e')

for xi, label in {0: '2/12', 4: '10/12'}.items():
    ax1.plot(xi, 0.714, 'r^', markersize=10, zorder=4)
    ax1.text(xi, 0.719, label, ha='center', fontsize=9,
             color='#d62728', fontweight='bold')

ax1.annotate('Best MF1: 0.676', xy=(3, 0.676), xytext=(1.8, 0.695),
             arrowprops=dict(arrowstyle='->', color='#1f77b4'),
             fontsize=9, color='#1f77b4')
ax1.annotate('Best Retroviral:\n10/12, AUC=0.833', xy=(4, 0.650),
             xytext=(2.9, 0.620),
             arrowprops=dict(arrowstyle='->', color='#2ca02c'),
             fontsize=9, color='#2ca02c')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1r.get_legend_handles_labels()
tp_patch = mpatches.Patch(color='#d62728', label='Retroviral TP count')
ax1.legend(lines1 + lines2 + [tp_patch],
           labels1 + labels2 + ['Retroviral TP count'],
           loc='lower left', fontsize=9)

ax2.axis('off')
key_changes = [
    "Baseline: RF +\nhandcrafted features,\nno ESM-2",
    "First ESM-2 integration;\nvoting ensemble",
    "Isotonic calibration +\npercentile thresholds",
    "SVM ensemble +\nMI-30 feature selection",
    "7-model rank-blend;\ncatalytic-window ESM-2;\nRNaseH domain features",
]
x_positions = np.linspace(0.07, 0.93, len(stages))
colours_strip = ['#555555', '#555555', '#555555', '#555555', '#2ca02c']
for xi, txt, col in zip(x_positions, key_changes, colours_strip):
    ax2.text(xi, 0.9, txt, ha='center', va='top', fontsize=7.5,
             color=col, transform=ax2.transAxes,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f8f8',
                       edgecolor='#cccccc', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.93])
out1 = OUT / '01_era1_progression.png'
plt.savefig(out1, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {out1}  ({out1.stat().st_size/1024:.0f} KB)")

# ── FIGURE 2: Era 2 (v8–v14) ────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9),
                                gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle("Era 2 Development: Optimising for CLS Metric (v8–v14)",
             fontsize=13, fontweight='bold', y=0.98)
fig.text(0.5, 0.935,
    "Metric redefinition at v8 revealed fundamental ranking failure (WSpearman\u22480 for binary classifiers)",
    ha='center', fontsize=10, color='#444444')

versions = ['v8', 'v8t', 'v9', 'v10', 'v11', 'v12/v13', 'v14']
x        = np.arange(len(versions))
cls_vals = [0.487, 0.613, 0.677, 0.776, 0.791, 0.792, 0.803]
pr_auc   = [0.563, 0.765, 0.727, 0.768, 0.799, 0.800, 0.811]
wsp_vals = [0.429, 0.433, 0.633, 0.798, 0.783, 0.784, 0.796]

ax1.axvspan(5.6, 6.4, alpha=0.13, color='#2ca02c', zorder=0)
ax1.axvline(x=2.5, color='grey', linestyle='--', linewidth=1.2, zorder=1)
ax1.text(2.54, 0.37, "LR replaces\ntree classifier", fontsize=8,
         color='grey', rotation=90, va='bottom')

ax1.plot(x, cls_vals, 'k-o', linewidth=2.5, markersize=8, label='CLS', zorder=4)
ax1.plot(x, pr_auc,   '--s', color='#1f77b4', linewidth=2, markersize=7,
         label='PR-AUC', zorder=3)
ax1.plot(x, wsp_vals, ':^',  color='#ff7f0e', linewidth=2, markersize=7,
         label='Weighted Spearman', zorder=3)

ax1.set_ylim(0.35, 0.87)
ax1.set_xticks(x)
ax1.set_xticklabels(versions, fontsize=11)
ax1.set_ylabel('Score', fontsize=11)
ax1.legend(loc='upper left', fontsize=10)

ax1.annotate(
    'PCA leakage fix\n(+0.099 CLS, largest single gain)',
    xy=(3, cls_vals[3]), xytext=(3.35, 0.71),
    arrowprops=dict(arrowstyle='->', color='#d62728', lw=2),
    fontsize=9, color='#d62728',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff0f0',
              edgecolor='#d62728', alpha=0.9))

ax1.annotate(
    'SUBMITTED\nCLS=0.803',
    xy=(6, cls_vals[6]), xytext=(4.9, 0.830),
    arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5),
    fontsize=9, color='#2ca02c', fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0fff0',
              edgecolor='#2ca02c', alpha=0.9))

cat_colours = {
    'Feature Engineering':   '#1f77b4',
    'Model Architecture':    '#ff7f0e',
    'Bug Fix / Leakage':     '#d62728',
    'Hyperparameter Tuning': '#7f7f7f',
}
cat_order = ['Feature Engineering', 'Model Architecture',
             'Bug Fix / Leakage', 'Hyperparameter Tuning']
categories = {
    'v8':      {'Model Architecture': 1.0},
    'v8t':     {'Feature Engineering': 1.0},
    'v9':      {'Feature Engineering': 1.0},
    'v10':     {'Bug Fix / Leakage': 0.6, 'Model Architecture': 0.4},
    'v11':     {'Feature Engineering': 1.0},
    'v12/v13': {'Feature Engineering': 1.0},
    'v14':     {'Feature Engineering': 0.6, 'Hyperparameter Tuning': 0.4},
}

bottoms = np.zeros(len(versions))
handles = []
for cat in cat_order:
    vals = np.array([categories[v].get(cat, 0.0) for v in versions])
    ax2.bar(x, vals, bottom=bottoms, color=cat_colours[cat], label=cat, width=0.6)
    bottoms += vals
    handles.append(mpatches.Patch(color=cat_colours[cat], label=cat))

ax2.set_xticks(x)
ax2.set_xticklabels(versions, fontsize=11)
ax2.set_yticks([])
ax2.set_ylabel('Change type', fontsize=9)
ax2.legend(handles=handles, loc='upper right', fontsize=8, ncol=2)
ax2.set_ylim(0, 1.25)

plt.tight_layout(rect=[0, 0, 1, 0.93])
out2 = OUT / '02_era2_progression.png'
plt.savefig(out2, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {out2}  ({out2.stat().st_size/1024:.0f} KB)")
