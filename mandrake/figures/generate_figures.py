import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('default')

OUT_DIR = r"C:\Users\yatin\Downloads\mandrake_data_25_03\mandrake_data\mandrake_github\figures"

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Era 1 Progression
# ─────────────────────────────────────────────────────────────────────────────

stages    = ['v1', 'v3', 'v4', 'v5', 'v6/v7']
mf1       = [0.634, 0.634, 0.607, 0.676, 0.650]
retro_auc = [0.627, None, None, None, 0.833]
retro_tp  = [2,    2,    None, None, 10]

x = np.arange(len(stages))

fig1, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 8),
    gridspec_kw={'height_ratios': [3, 1]},
    facecolor='white'
)
fig1.patch.set_facecolor('white')

# ── Panel 1 ──────────────────────────────────────────────────────────────────
ax1.set_facecolor('white')

# Left y-axis: Macro-F1
line_mf1, = ax1.plot(x, mf1, color='steelblue', linestyle='-',
                     marker='o', linewidth=2, markersize=8, label='Macro-F1')
ax1.set_ylim(0.55, 0.72)
ax1.set_ylabel('Macro-F1', color='steelblue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_xticks(x)
ax1.set_xticklabels(stages, fontsize=11)
ax1.set_xlim(-0.5, len(stages) - 0.5)

# Right y-axis: Retroviral AUC
ax1r = ax1.twinx()
auc_x = [xi for xi, v in zip(x, retro_auc) if v is not None]
auc_y = [v for v in retro_auc if v is not None]
line_auc, = ax1r.plot(auc_x, auc_y, color='darkorange', linestyle='--',
                      marker='s', linewidth=2, markersize=8, label='Retroviral AUC')
ax1r.set_ylim(0.5, 0.95)
ax1r.set_ylabel('Retroviral AUC', color='darkorange', fontsize=12)
ax1r.tick_params(axis='y', labelcolor='darkorange')

# Retroviral TP markers (above plot area on left-axis scale)
# v1 (index 0): TP=2, v6/v7 (index 4): TP=10  — skip v3 (index 1) per spec
tp_plot_indices = [0, 4]
tp_labels       = ['2/12', '10/12']
tp_y            = 0.715   # just above the top of left-axis range
line_tp, = ax1.plot(tp_plot_indices, [tp_y, tp_y], color='red',
                    marker='o', linestyle='none', markersize=9,
                    label='Retroviral TP', clip_on=False, zorder=5)
for xi, lbl in zip(tp_plot_indices, tp_labels):
    ax1.text(xi, tp_y + 0.003, lbl, ha='center', va='bottom',
             color='red', fontsize=9, clip_on=False)

# Green shading for v6/v7 region
ax1.axvspan(3.5, 4.5, alpha=0.12, color='green', zorder=0)

# Vertical dashed line between v5 (x=3) and v6/v7 (x=4)
vline_x = 3.5
ax1.axvline(x=vline_x, color='grey', linestyle='--', linewidth=1.2, zorder=1)
ax1.text(vline_x - 0.06, 0.57, 'Rank averaging\nintroduced',
         rotation=90, va='bottom', ha='right', color='grey', fontsize=9)

# Annotate v5 best MF1
ax1.annotate('Best MF1: 0.676',
             xy=(3, 0.676), xytext=(2.2, 0.695),
             arrowprops=dict(arrowstyle='->', color='steelblue'),
             fontsize=10, color='steelblue',
             ha='center')

# Annotate v6/v7 best Retroviral
ax1.annotate('Best Retroviral:\n10/12, AUC=0.833',
             xy=(4, 0.650), xytext=(3.1, 0.700),
             arrowprops=dict(arrowstyle='->', color='darkorange'),
             fontsize=10, color='darkorange',
             ha='center')

# Legend (combined)
handles = [line_mf1, line_auc, line_tp]
ax1.legend(handles=handles, loc='lower left', fontsize=10,
           framealpha=0.9)

ax1.set_title(
    'Era 1 Development: Optimising for Macro-F1 and Retroviral Coverage (v1–v7)',
    fontsize=13, fontweight='bold', pad=12
)
ax1.text(0.5, 1.02,
         'Final submission: MF1=0.650 | Retroviral AUC=0.833 | 10/12 Retroviral TPs',
         transform=ax1.transAxes, ha='center', va='bottom',
         fontsize=10, color='dimgrey', style='italic')

# ── Panel 2: annotation strip ─────────────────────────────────────────────
ax2.axis('off')
ax2.set_facecolor('white')

stage_annotations = {
    'v1':    ("Baseline: RF + handcrafted features, no ESM-2", 'steelblue'),
    'v3':    ("First ESM-2 integration; voting ensemble", 'steelblue'),
    'v4':    ("Isotonic calibration + percentile thresholds", 'steelblue'),
    'v5':    ("SVM ensemble + MI-30 feature selection", 'steelblue'),
    'v6/v7': ("7-model rank-blend; catalytic-window ESM-2;\nRNaseH domain features",
              'darkorange'),
}

# Map stage name -> normalised x position in [0,1]
total_stages = len(stages)
for xi, stage in enumerate(stages):
    norm_x = xi / (total_stages - 1)          # 0 to 1
    text, colour = stage_annotations[stage]
    ax2.text(norm_x, 0.7, stage, ha='center', va='top',
             fontsize=9, fontweight='bold', color=colour,
             transform=ax2.transAxes)
    ax2.text(norm_x, 0.45, text, ha='center', va='top',
             fontsize=8, color=colour,
             transform=ax2.transAxes, wrap=True)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, '01_era1_progression.png')
fig1.savefig(out1, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print(f"Saved: {out1}  ({os.path.getsize(out1):,} bytes)")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Era 2 Progression
# ─────────────────────────────────────────────────────────────────────────────

versions = ['v8', 'v8t', 'v9', 'v10', 'v11', 'v12/v13', 'v14']
cls_vals = [0.487, 0.613, 0.677, 0.776, 0.791, 0.792, 0.803]
pr_auc   = [0.563, 0.765, 0.727, 0.768, 0.799, 0.800, 0.811]
wsp_vals = [0.429, 0.433, 0.633, 0.798, 0.783, 0.784, 0.796]

x2 = np.arange(len(versions))

categories = {
    'v8':      {'Model Architecture': 1.0},
    'v8t':     {'Feature Engineering': 1.0},
    'v9':      {'Feature Engineering': 1.0},
    'v10':     {'Bug Fix / Leakage': 0.6, 'Model Architecture': 0.4},
    'v11':     {'Feature Engineering': 1.0},
    'v12/v13': {'Feature Engineering': 1.0},
    'v14':     {'Feature Engineering': 0.6, 'Hyperparameter Tuning': 0.4},
}
colours = {
    'Feature Engineering':   '#1f77b4',
    'Model Architecture':    '#ff7f0e',
    'Bug Fix / Leakage':     '#d62728',
    'Hyperparameter Tuning': '#7f7f7f',
}
cat_order = ['Feature Engineering', 'Model Architecture',
             'Bug Fix / Leakage', 'Hyperparameter Tuning']

fig2, (ax3, ax4) = plt.subplots(
    2, 1,
    figsize=(13, 9),
    gridspec_kw={'height_ratios': [3, 1]},
    facecolor='white'
)
fig2.patch.set_facecolor('white')

# ── Panel 1: line chart ───────────────────────────────────────────────────
ax3.set_facecolor('white')
ax3.set_ylim(0.35, 0.87)
ax3.set_xlim(-0.5, len(versions) - 0.5)
ax3.set_xticks(x2)
ax3.set_xticklabels(versions, fontsize=11)

line_cls, = ax3.plot(x2, cls_vals, color='black', linestyle='-',
                     marker='o', linewidth=2, markersize=8, label='CLS')
line_pr,  = ax3.plot(x2, pr_auc,   color='#1f77b4', linestyle='--',
                     marker='s', linewidth=2, markersize=8, label='PR-AUC')
line_wsp, = ax3.plot(x2, wsp_vals, color='#ff7f0e', linestyle=':',
                     marker='^', linewidth=2, markersize=8, label='W.Spearman')

# Green shading around v14 (index 6)
ax3.axvspan(5.7, 6.3, alpha=0.15, color='green')

# Vertical dashed line between v9 (x=2) and v10 (x=3)
ax3.axvline(x=2.5, color='grey', linestyle='--', linewidth=1.2)
ax3.text(2.44, 0.37, 'LR replaces\ntree classifier',
         rotation=90, va='bottom', ha='right', color='grey', fontsize=9)

# Red upward arrow at v10 (index 3)
ax3.annotate('PCA leakage fix\n(+0.099 CLS, largest single gain)',
             xy=(3, cls_vals[3]), xytext=(3, 0.58),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.8),
             fontsize=9, color='red', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', fc='#fff0f0', ec='red', alpha=0.8))

# Annotate v14 submitted
ax3.annotate('SUBMITTED\nCLS=0.803',
             xy=(6, cls_vals[6]), xytext=(5.1, 0.820),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
             fontsize=10, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', ec='black', alpha=0.9))

ax3.set_ylabel('Score', fontsize=12)
ax3.legend(handles=[line_cls, line_pr, line_wsp], loc='upper left',
           fontsize=10, framealpha=0.9)

ax3.set_title(
    'Era 2 Development: Optimising for CLS Metric (v8–v14)',
    fontsize=13, fontweight='bold', pad=12
)
ax3.text(0.5, 1.02,
         'Metric redefinition at v8 revealed fundamental ranking failure (WSpearman≈0 for binary classifiers)',
         transform=ax3.transAxes, ha='center', va='bottom',
         fontsize=10, color='dimgrey', style='italic')

# ── Panel 2: stacked bar chart ────────────────────────────────────────────
ax4.set_facecolor('white')

bottoms = np.zeros(len(versions))
bars_for_legend = {}

for cat in cat_order:
    heights = []
    for ver in versions:
        h = categories[ver].get(cat, 0.0)
        heights.append(h)
    heights = np.array(heights)
    bar = ax4.bar(x2, heights, bottom=bottoms,
                  color=colours[cat], label=cat, width=0.6)
    bottoms += heights
    if any(h > 0 for h in heights):
        bars_for_legend[cat] = bar

ax4.set_xticks(x2)
ax4.set_xticklabels(versions, fontsize=11)
ax4.set_ylim(0, 1)
ax4.set_ylabel('Change category', fontsize=11)
ax4.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax4.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.0'])

legend_patches = [mpatches.Patch(color=colours[cat], label=cat)
                  for cat in cat_order if cat in bars_for_legend]
ax4.legend(handles=legend_patches, loc='upper right', fontsize=9,
           framealpha=0.9)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, '02_era2_progression.png')
fig2.savefig(out2, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f"Saved: {out2}  ({os.path.getsize(out2):,} bytes)")
