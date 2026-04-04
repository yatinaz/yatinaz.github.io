#!/usr/bin/env python3
"""
eeg_pipeline.py  —  EEG Motor Imagery Classification Pipeline
======================================================
Dataset : PhysioNet EEGBCI (mne.datasets.eegbci)
Task    : Left-hand vs right-hand motor imagery
Subjects: 1-10  |  Runs: 6, 10, 14 (imagination only)

Pipeline
--------
1.  Load & concatenate EDF runs per subject
2.  Set standard_1020 montage (electrode positions)
3.  Bandpass filter 1-40 Hz (FIR, Hamming window)
4.  Average re-reference
5.  ICA (FastICA, 20 components) — remove up to 2 EOG components
6.  Epoch −1 … +4 s around T1 (left) / T2 (right) cues
7.  Baseline correction (−1 … 0 s)
8.  Band-power features via Welch PSD (delta/theta/alpha/beta/gamma)
9.  Time-frequency representation for C3 & C4 (Morlet wavelets)
10. LDA + RBF-SVM, stratified 5-fold cross-validation
11. Export five JSON files for the interactive HTML visualisation

All random seeds are fixed for full reproducibility.

Usage
-----
    pip install mne scikit-learn numpy
    python eeg_pipeline.py
"""

import os
import json
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# ── MNE ─────────────────────────────────────────────────────────────────────
import mne
from mne.datasets import eegbci
from mne.time_frequency import psd_array_welch

mne.set_log_level('WARNING')

# ── scikit-learn ─────────────────────────────────────────────────────────────
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from mne.decoding import CSP
from sklearn.base import clone
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM

# ── REPRODUCIBILITY ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── CONFIGURATION ────────────────────────────────────────────────────────────
SUBJECTS   = list(range(1, 11))   # subjects 1-10
RUNS       = [6, 10, 14]          # motor imagery: T1=left fist, T2=right fist
L_FREQ     = 1.0                  # bandpass low-cut (Hz)  — removes DC drift
H_FREQ     = 40.0                 # bandpass high-cut (Hz) — removes EMG/noise
TMIN       = -1.0                 # epoch start relative to event (s)
TMAX       =  4.0                 # epoch end relative to event (s)
BASELINE   = (-1.0, 0.0)          # pre-cue baseline for correction (s)
N_ICA      = 20                   # ICA components to estimate
SRATE_OUT  = 100                  # target sample rate for time-series JSON (Hz)
OUTPUT_DIR = 'outputs'

# Frequency bands for band-power features
BANDS = {
    'delta': ( 1.0,  4.0),
    'theta': ( 4.0,  8.0),
    'alpha': ( 8.0, 13.0),   # primary ERD/ERS band for motor imagery
    'beta' : (13.0, 30.0),   # secondary ERD band
    'gamma': (30.0, 40.0),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── UTILITY FUNCTIONS ────────────────────────────────────────────────────────

def round_list(arr, d=4):
    """Convert numpy array to rounded Python list for compact JSON."""
    return [round(float(v), d) for v in np.asarray(arr).ravel()]


def round_2d(arr2d, d=3):
    """Convert 2-D array to nested list with rounded values."""
    return [[round(float(v), d) for v in row] for row in arr2d]


def cart_to_2d(x, y, z):
    """
    Project a 3-D head-surface point to a 2-D disc using azimuthal-
    equidistant projection (the same geometry MNE uses for topomaps).

    MNE coordinate convention: nose → +y, right ear → +x, vertex → +z.
    Returns (x2d, y2d) scaled so the ear-level ring maps to radius 1.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    if r < 1e-9:
        return 0.0, 0.0
    xn, yn, zn = x / r, y / r, z / r
    phi   = np.arccos(np.clip(zn, -1.0, 1.0))   # polar angle from vertex
    theta = np.arctan2(yn, xn)                    # azimuth around z-axis
    rho   = phi / (np.pi / 2)                     # normalise to unit circle
    return float(rho * np.cos(theta)), float(rho * np.sin(theta))


# ── LOADING & PREPROCESSING ──────────────────────────────────────────────────

def load_subject(subject):
    """
    Download (once) and load all motor-imagery EDF files for one subject.
    Concatenates runs 6, 10, 14 into a single continuous Raw object.
    """
    fnames = eegbci.load_data(subject, RUNS, verbose=False, update_path=True)
    raws = [mne.io.read_raw_edf(f, preload=True, verbose=False)
            for f in fnames]
    raw = mne.concatenate_raws(raws)
    # Rename e.g. 'Fc5..' → 'FC5' to match standard 10-20 labels
    eegbci.standardize(raw)
    return raw


def preprocess(raw):
    """
    Apply electrode montage, bandpass filter, and average re-reference
    to the continuous raw signal.
    """
    # Set standard 10-20 positions (needed for ICA EOG detection & topomap)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore', verbose=False)

    # FIR bandpass 1-40 Hz with Hamming window
    # - Low-cut removes slow physiological drift and DC offset
    # - High-cut removes EMG artefacts and power-line noise above our bands
    raw.filter(L_FREQ, H_FREQ, fir_window='hamming', verbose=False)

    # Average reference: subtracts the mean signal across all electrodes
    # from each channel, reducing the influence of reference electrode choice
    raw.set_eeg_reference('average', projection=True, verbose=False)
    raw.apply_proj()

    return raw


def run_ica(raw):
    """
    Fit Independent Component Analysis to identify and remove ocular
    artefacts (eye blinks, saccades).

    Because the EEGBCI dataset has no dedicated EOG channel, we use
    frontal EEG electrodes (Fp1, Fp2) as surrogates.  Only components
    with a frontal correlation > 0.25 are excluded (up to 2 maximum).

    Returns: (cleaned_raw, n_components_excluded)
    """
    ica = mne.preprocessing.ICA(
        n_components=N_ICA, method='fastica',
        max_iter=1000, random_state=SEED
    )
    ica.fit(raw, verbose=False)

    # Identify EOG-correlated components via frontal-channel correlation
    eog_chs = [c for c in ['Fp1', 'Fp2'] if c in raw.ch_names]
    excluded = []
    if eog_chs:
        try:
            inds, scores = ica.find_bads_eog(
                raw, ch_name=eog_chs, verbose=False)
            # Keep only components with clear frontal correlation
            excluded = [i for i, s in zip(inds, np.abs(scores[inds]))
                        if s > 0.25][:2]
        except Exception:
            pass

    ica.exclude = excluded
    cleaned = raw.copy()
    ica.apply(cleaned, verbose=False)
    return cleaned, len(excluded)


def make_epochs(raw):
    """
    Convert MNE annotations to events and extract fixed-length epochs.

    Event mapping in motor-imagery runs (6, 10, 14):
      T1 (code 2) → left-hand motor imagery
      T2 (code 3) → right-hand motor imagery

    Returns an Epochs object or None if no events found.
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    id_map = {}
    if 'T1' in event_id:
        id_map['left']  = event_id['T1']
    if 'T2' in event_id:
        id_map['right'] = event_id['T2']
    if not id_map:
        return None

    picks  = mne.pick_types(raw.info, eeg=True, exclude='bads')
    epochs = mne.Epochs(
        raw, events, event_id=id_map,
        tmin=TMIN, tmax=TMAX, baseline=BASELINE,
        picks=picks, preload=True,
        reject_by_annotation=True, verbose=False
    )
    return epochs


# ── BAND POWER ───────────────────────────────────────────────────────────────

def compute_band_power(epochs):
    """
    Estimate mean band power in each frequency band for every epoch and
    channel using Welch's method (2-second FFT window).

    Returns a dict mapping band name → ndarray of shape (n_epochs, n_channels).
    """
    data  = epochs.get_data()           # (n_epochs, n_ch, n_times)
    sfreq = epochs.info['sfreq']
    n_fft = int(sfreq * 2)             # 2 s window gives ~0.5 Hz resolution

    bp = {}
    for band, (fmin, fmax) in BANDS.items():
        psds, _ = psd_array_welch(
            data, sfreq=sfreq, fmin=fmin, fmax=fmax,
            n_fft=n_fft, verbose=False
        )                               # (n_epochs, n_ch, n_freqs)
        bp[band] = psds.mean(axis=-1)  # average over frequency → (n_ep, n_ch)
    return bp


# ── ELECTRODE POSITIONS ───────────────────────────────────────────────────────

def get_2d_positions(ch_names):
    """
    Map channel names to 2-D scalp coordinates using the standard_1020
    montage and azimuthal-equidistant projection.

    Returns (valid_channel_names, x_positions, y_positions).
    """
    montage   = mne.channels.make_standard_montage('standard_1020')
    ch_pos_3d = montage.get_positions()['ch_pos']

    valid_chs, pos_x, pos_y = [], [], []
    for ch in ch_names:
        if ch in ch_pos_3d:
            x2d, y2d = cart_to_2d(*ch_pos_3d[ch])
            valid_chs.append(ch)
            pos_x.append(x2d)
            pos_y.append(y2d)

    return valid_chs, pos_x, pos_y


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE LOOP — process all subjects
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  EEG Motor Imagery Pipeline")
print(f"  Subjects: {SUBJECTS[0]}-{SUBJECTS[-1]}  |  Runs: {RUNS}")
print("=" * 60)

all_bp          = []   # band-power dict per subject
all_labels      = []   # integer label array per subject (2=left, 3=right)
all_epochs_data = []   # narrowband epoch data for CSP (8-30 Hz)
all_ch          = None # channel names (set from first successful subject)

# Save full data from subject 1 for the visualisation figures
s1_raw    = None
s1_clean  = None
s1_epochs = None

for subj in SUBJECTS:
    print(f"\nSubject {subj:2d}/{len(SUBJECTS)}")
    try:
        raw = load_subject(subj)

        # Keep truly unprocessed copy of subject 1 BEFORE filter/ref/ICA
        # This gives a visually distinct baseline for Figure 1
        if subj == 1:
            s1_raw = raw.copy()

        raw = preprocess(raw)
        cleaned, n_ex = run_ica(raw)
        print(f"  ICA: removed {n_ex} component(s)")

        if subj == 1:
            s1_clean = cleaned.copy()

        epochs = make_epochs(cleaned)
        if epochs is None:
            print("  No T1/T2 events found — skipping")
            continue

        if subj == 1:
            s1_epochs = epochs.copy()

        bp     = compute_band_power(epochs)
        labels = epochs.events[:, 2]   # 2 = left, 3 = right

        if all_ch is None:
            all_ch = list(epochs.ch_names)

        # Narrowband filter 8-30 Hz for CSP — fitted within CV to prevent leakage
        ep_mb = epochs.copy().filter(8., 30., verbose=False)
        all_epochs_data.append(ep_mb.get_data())   # (n_ep, n_ch, n_times)

        all_bp.append(bp)
        all_labels.append(labels)

        n_left  = int((labels == 2).sum())
        n_right = int((labels == 3).sum())
        print(f"  Epochs: {len(epochs)} total  ({n_left} left / {n_right} right)")

    except Exception as exc:
        print(f"  ERROR: {exc}")
        continue

print(f"\nFinished loading. {len(all_bp)} subjects contributed data.")

# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE MATRIX — labels shared by both approaches
# ═════════════════════════════════════════════════════════════════════════════

y_parts = [(lab_s == 3).astype(int) for lab_s in all_labels]
y = np.concatenate(y_parts)   # 0=left, 1=right

# Band-power matrix (kept for figure exports)
FEAT_BANDS = ['alpha', 'beta']
X_bp = np.vstack([np.hstack([bp_s[b] for b in FEAT_BANDS]) for bp_s in all_bp])

# CSP epoch matrix: (total_epochs, n_channels, n_times)
X_csp = np.vstack(all_epochs_data)

print(f"\nEpoch matrix for CSP: {X_csp.shape}  "
      f"left={int((y==0).sum())}  right={int((y==1).sum())}")

# ═════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION — CSP + LDA  &  CSP + RBF-SVM,  stratified 5-fold CV
#
#  CSP (Common Spatial Patterns) learns 6 spatial filters that maximally
#  separate the covariance structure of left vs right epochs.  The log-variance
#  of the filtered signals forms a compact, highly discriminative feature vector.
#  Crucially, CSP is fitted only on training data each fold to prevent leakage.
# ═════════════════════════════════════════════════════════════════════════════

classifiers = {
    'lda': Pipeline([
        ('csp', CSP(n_components=6, log=True, reg='ledoit_wolf')),
        ('clf', LDA())
    ]),
    'svm': Pipeline([
        ('csp', CSP(n_components=6, log=True, reg='ledoit_wolf')),
        ('sc',  StandardScaler()),
        ('clf', SVC(kernel='rbf', C=10., gamma='scale',
                    probability=True, random_state=SEED))
    ])
}

# Within-subject CV: fit CSP on each subject's own training folds.
# CSP spatial filters are subject-specific; pooling subjects dilutes them.
# 10 subjects × 5 folds = 50 accuracy values; report per-subject mean.
results = {name: {'fold_accs': [], 'y_true': [], 'y_score': []}
           for name in classifiers}
subj_accs = {name: [] for name in classifiers}

print("\nWithin-subject 5-fold CV (CSP+classifier):")
skf_subj = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Epoch data and labels split per subject
subj_sizes = [len(lab) for lab in y_parts]
X_splits = np.split(X_csp, np.cumsum(subj_sizes[:-1]))
y_splits = np.split(y,     np.cumsum(subj_sizes[:-1]))

for s_idx, (Xs, ys) in enumerate(zip(X_splits, y_splits), 1):
    s_fold_accs = {name: [] for name in classifiers}
    for tr, te in skf_subj.split(Xs, ys):
        Xtr, Xte = Xs[tr], Xs[te]
        ytr, yte = ys[tr], ys[te]
        for name, clf in classifiers.items():
            c = clone(clf)
            c.fit(Xtr, ytr)
            prob = c.predict_proba(Xte)[:, 1]
            pred = c.predict(Xte)
            s_fold_accs[name].append(float((pred == yte).mean()))
            results[name]['y_true'].extend(yte.tolist())
            results[name]['y_score'].extend(prob.tolist())
    for name in classifiers:
        sm = float(np.mean(s_fold_accs[name]))
        subj_accs[name].append(sm)
        results[name]['fold_accs'].append(sm)
    print(f"  Subj {s_idx:2d}:  CSP+LDA={subj_accs['lda'][-1]:.3f}  "
          f"CSP+SVM={subj_accs['svm'][-1]:.3f}")

# ═════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION — Riemannian MDM (within-subject)
#
#  Each epoch's 64×64 covariance matrix (estimated with Ledoit-Wolf shrinkage)
#  is a point on the symmetric positive definite (SPD) manifold.  MDM classifies
#  by computing the Riemannian mean of each class and assigning the epoch to the
#  nearest mean under the affine-invariant metric — no hyperparameters to tune.
# ═════════════════════════════════════════════════════════════════════════════

results['riemann'] = {'fold_accs': [], 'y_true': [], 'y_score': []}
cov_est = Covariances(estimator='lwf')  # Ledoit-Wolf-Fre hat covariance

print("\nWithin-subject 5-fold CV (Riemannian MDM):")
for s_idx, (Xs, ys) in enumerate(zip(X_splits, y_splits), 1):
    covs = cov_est.fit_transform(Xs)    # (n_ep, 64, 64) SPD matrices
    s_fold_accs_r = []
    for tr, te in skf_subj.split(covs, ys):
        mdm = MDM(metric='riemann')
        mdm.fit(covs[tr], ys[tr])
        pred = mdm.predict(covs[te])
        prob = mdm.predict_proba(covs[te])[:, 1]
        s_fold_accs_r.append(float((pred == ys[te]).mean()))
        results['riemann']['y_true'].extend(ys[te].tolist())
        results['riemann']['y_score'].extend(prob.tolist())
    sm = float(np.mean(s_fold_accs_r))
    results['riemann']['fold_accs'].append(sm)
    print(f"  Subj {s_idx:2d}:  MDM={sm:.3f}")

# ═════════════════════════════════════════════════════════════════════════════
#  JSON EXPORT
# ═════════════════════════════════════════════════════════════════════════════

print("\nExporting JSON files …")

# ── Figure 1: Raw vs ICA-cleaned time series ─────────────────────────────────

# Show key channels covering motor cortex and frontal (EOG artefact region)
KEY_CHS = [c for c in ['C3', 'Cz', 'C4', 'C5', 'C6', 'Fp1', 'Fp2', 'FC3', 'FC4']
           if c in s1_raw.ch_names]

# Crop to a 4-second window starting 10 s into the recording
T0, T1_END = 10.0, 14.0
sfreq_in   = s1_raw.info['sfreq']
decim      = max(1, int(sfreq_in / SRATE_OUT))  # decimation factor

raw_seg,   times = s1_raw[KEY_CHS,
    int(T0 * sfreq_in) : int(T1_END * sfreq_in)]
clean_seg, _     = s1_clean[KEY_CHS,
    int(T0 * sfreq_in) : int(T1_END * sfreq_in)]

# Decimate and convert V → µV
raw_seg   = raw_seg[:,   ::decim] * 1e6
clean_seg = clean_seg[:, ::decim] * 1e6
times_out = (times[::decim] - T0).tolist()

ts_json = {
    'sfreq'   : SRATE_OUT,
    'channels': KEY_CHS,
    'time'    : round_list(times_out),
    'raw'     : {ch: round_list(raw_seg[i])   for i, ch in enumerate(KEY_CHS)},
    'cleaned' : {ch: round_list(clean_seg[i]) for i, ch in enumerate(KEY_CHS)},
}
with open(f'{OUTPUT_DIR}/eeg_timeseries.json', 'w') as fh:
    json.dump(ts_json, fh, separators=(',', ':'))
print("  [OK]  eeg_timeseries.json")

# ── Figure 2: Power Spectral Density per condition ────────────────────────────

PSD_CHS = [c for c in ['C3', 'Cz', 'C4'] if c in s1_epochs.ch_names]

sfreq_ep = s1_epochs.info['sfreq']
ep_data  = s1_epochs.get_data()          # (n_ep, n_ch, n_times)
ep_codes = s1_epochs.events[:, 2]

left_psd  = {}
right_psd = {}

for ch in PSD_CHS:
    ch_idx = list(s1_epochs.ch_names).index(ch)

    for code, store in [(2, left_psd), (3, right_psd)]:
        mask   = ep_codes == code
        if mask.sum() == 0:
            store[ch] = {'mean': [], 'sem': []}
            continue
        # Compute full-spectrum PSD for this channel and condition
        ep_ch = ep_data[mask, ch_idx, :]          # (n_cond_epochs, n_times)
        psds, freqs = psd_array_welch(
            ep_ch, sfreq=sfreq_ep, fmin=1.0, fmax=40.0,
            n_fft=int(sfreq_ep * 4), verbose=False
        )                                          # (n_ep, n_freqs)
        mu  = psds.mean(axis=0) * 1e12   # V²/Hz → µV²/Hz
        sem = psds.std(axis=0)  * 1e12 / np.sqrt(psds.shape[0])
        store[ch] = {'mean': round_list(mu), 'sem': round_list(sem)}

psd_json = {
    'freqs'     : round_list(freqs),
    'channels'  : PSD_CHS,
    'left_hand' : left_psd,
    'right_hand': right_psd,
    'alpha_band': [8, 13],
    'beta_band' : [13, 30],
}
with open(f'{OUTPUT_DIR}/psd_data.json', 'w') as fh:
    json.dump(psd_json, fh, separators=(',', ':'))
print("  [OK]  psd_data.json")

# ── Figure 3: Alpha topomap (all channels, all subjects) ─────────────────────

# Pool alpha power across subjects; average per channel per condition
n_ch       = len(all_ch)
alpha_L    = np.zeros(n_ch)
alpha_R    = np.zeros(n_ch)
count_L    = np.zeros(n_ch)
count_R    = np.zeros(n_ch)

for bp_s, lab_s in zip(all_bp, all_labels):
    a = bp_s['alpha']               # (n_epochs, n_ch)
    for ep_i, code in enumerate(lab_s):
        if code == 2:               # left-hand
            alpha_L  += a[ep_i]
            count_L  += 1
        elif code == 3:             # right-hand
            alpha_R  += a[ep_i]
            count_R  += 1

# Avoid division by zero for channels missing in some subjects
count_L = np.where(count_L == 0, 1, count_L)
count_R = np.where(count_R == 0, 1, count_R)
alpha_L /= count_L
alpha_R /= count_R

# ERD (%) = how much alpha power changes right vs left condition
# Negative = right-hand imagery suppresses alpha (contralateral motor area)
erd = (alpha_R - alpha_L) / (alpha_L + 1e-10) * 100.0

valid_chs, pos_x, pos_y = get_2d_positions(all_ch)
valid_idx = [all_ch.index(c) for c in valid_chs]

topo_json = {
    'channels'        : valid_chs,
    'pos_x'           : [round(v, 4) for v in pos_x],
    'pos_y'           : [round(v, 4) for v in pos_y],
    'left_hand_alpha' : round_list(alpha_L[valid_idx] * 1e12),   # V²/Hz → µV²/Hz
    'right_hand_alpha': round_list(alpha_R[valid_idx] * 1e12),
    'erd_percent'     : round_list(erd[valid_idx]),               # ratio, unit-independent
}
with open(f'{OUTPUT_DIR}/topomap_data.json', 'w') as fh:
    json.dump(topo_json, fh, separators=(',', ':'))
print("  [OK]  topomap_data.json")

# ── Figure 4: Time-frequency representation (Morlet) for C3 and C4 ───────────

TFR_CHS   = [c for c in ['C3', 'C4'] if c in s1_epochs.ch_names]
TFR_FREQS = np.arange(4, 41, 2, dtype=float)   # 4-40 Hz in 2 Hz steps
n_cycles  = TFR_FREQS / 2.0                      # variable-cycle Morlet

ep_codes1 = s1_epochs.events[:, 2]
tfr_json  = {'freqs': TFR_FREQS.tolist(), 'channels': TFR_CHS}

for ch in TFR_CHS:
    if ch not in s1_epochs.ch_names:
        continue
    ch_idx = list(s1_epochs.ch_names).index(ch)

    for cond_name, cond_code in [('left', 2), ('right', 3)]:
        mask  = ep_codes1 == cond_code
        if mask.sum() == 0:
            continue

        # Extract single-channel epoch data: (n_ep, 1, n_times)
        ep_ch = s1_epochs.get_data()[mask, ch_idx : ch_idx + 1, :]

        # Build a single-channel EpochsArray for tfr_morlet
        info_1ch = mne.create_info(
            [ch], s1_epochs.info['sfreq'], ch_types='eeg')
        ep_arr   = mne.EpochsArray(ep_ch, info_1ch, verbose=False)

        # Morlet wavelet convolution — decim=4 keeps JSON manageable
        power = mne.time_frequency.tfr_morlet(
            ep_arr, freqs=TFR_FREQS, n_cycles=n_cycles,
            return_itc=False, average=True, decim=4, verbose=False
        )                                     # power.data: (1, n_freqs, n_times)

        times_tfr = power.times
        data_2d   = power.data[0]             # (n_freqs, n_times)

        # Convert to dB relative to pre-cue baseline (−1 … 0 s)
        # A negative dB value indicates event-related desynchronisation (ERD)
        base_mask     = (times_tfr >= BASELINE[0]) & (times_tfr <= BASELINE[1])
        baseline_mean = data_2d[:, base_mask].mean(axis=1, keepdims=True)
        data_db       = 10 * np.log10(data_2d / (baseline_mean + 1e-30))

        tfr_json[f'{ch}_{cond_name}'] = round_2d(data_db, 2)

tfr_json['times'] = round_list(times_tfr)
with open(f'{OUTPUT_DIR}/tfr_data.json', 'w') as fh:
    json.dump(tfr_json, fh, separators=(',', ':'))
print("  [OK]  tfr_data.json")

# ── Figure 5: Classification results ─────────────────────────────────────────

clf_json = {}
for name in ['lda', 'svm', 'riemann']:
    res = results[name]
    yt  = np.array(res['y_true'])
    ys  = np.array(res['y_score'])

    cm             = confusion_matrix(yt, (ys > 0.5).astype(int))
    fpr, tpr, thr  = roc_curve(yt, ys)
    auc            = roc_auc_score(yt, ys)
    accs           = res['fold_accs']

    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))
    label = {'lda':'CSP+LDA','svm':'CSP+SVM','riemann':'MDM (Riemann)'}.get(name,name)
    print(f"\n  {label:7s}  acc={mean_acc:.3f}+/-{std_acc:.3f}  AUC={auc:.3f}")
    print(f"       confusion matrix: {cm.tolist()}")

    clf_json[name] = {
        'fold_accs'       : [round(a, 4) for a in accs],
        'mean_acc'        : round(mean_acc, 4),
        'std_acc'         : round(std_acc,  4),
        'auc'             : round(float(auc), 4),
        'confusion_matrix': cm.tolist(),
        'roc': {
            'fpr'       : round_list(fpr),
            'tpr'       : round_list(tpr),
            'thresholds': round_list(np.clip(thr, 0, 1)),
        },
    }

with open(f'{OUTPUT_DIR}/classification_results.json', 'w') as fh:
    json.dump(clf_json, fh, separators=(',', ':'))
print("  [OK]  classification_results.json")

print(f"\n{'='*60}")
print(f"  All done! Files are in ./{OUTPUT_DIR}/")
print(f"  Open index.html in a browser to view the interactive page.")
print(f"{'='*60}\n")
