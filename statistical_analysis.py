"""
AMR vs Mass Spectra Statistical Analysis
Species: Staphylococcus epidermidis
Target drugs: Top 10 from AMR proportion chart
Test: Mann-Whitney U test (S group vs R group)
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. CONFIGURATION — adjust paths here
# ─────────────────────────────────────────────
AMR_CSV_PATH    = r"C:\Users\User\Downloads\filtered_A_S_epidermidis\filtered_A_S_epidermidis\2018_S_epidermidis_metadata.csv"          # your full 2554-row TSV/CSV
SPECTRA_ROOT    = r"C:\Users\User\Downloads\filtered_A_S_epidermidis\filtered_A_S_epidermidis\preprocessed"
OUTPUT_DIR      = "amr_spectra_results"
TARGET_SPECIES  = "Staphylococcus epidermidis"

# Top 10 drugs from the AMR proportion chart (highest resistance first)
TARGET_DRUGS = [
    "Ampicillin-Amoxicillin",
    "Penicillin",
    "Piperacillin-Tazobactam",
    "Ceftriaxone",
    "Meropenem",
    "Amoxicillin-Clavulanic acid",
    "Oxacillin",
    "Fusidic acid",
    "Ciprofloxacin",
    "Cotrimoxazole",
]

# Mass range for feature extraction (common MALDI-TOF range)
MASS_MIN = 2000
MASS_MAX = 20000
N_BINS   = 500   # interpolation bins for alignment

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD AMR DATA
# ─────────────────────────────────────────────
def load_amr(path):
    """Load AMR table. Use full code column (e.g. UUID_MALDI1) as the key."""
    sep = '\t' if path.endswith('.tsv') else ','
    df = pd.read_csv(path, sep=sep, low_memory=False)

    # Keep only S. epidermidis
    df = df[df['species'].str.strip() == TARGET_SPECIES].copy()
    print(f"[AMR] {len(df)} rows for {TARGET_SPECIES}")

    # Use full code as key (matches filename without .txt extension)
    # e.g. "fe6fc4ee-6a4b-4b51-a4f2-e07812e935ac_MALDI1"
    df['uuid'] = df['code'].str.strip()
    df = df.dropna(subset=['uuid'])

    # Keep only relevant columns
    available_drugs = [d for d in TARGET_DRUGS if d in df.columns]
    missing = [d for d in TARGET_DRUGS if d not in df.columns]
    if missing:
        print(f"[AMR] WARNING — drugs not found in columns: {missing}")

    return df[['uuid'] + available_drugs]

# ─────────────────────────────────────────────
# 2. FIND + PARSE SPECTRA FILES
# ─────────────────────────────────────────────
def find_spectra_files(root):
    """
    Recursively find all spectrum text files.
    Expected filename pattern: contains UUID in path.
    Files contain two columns: mass  intensity
    with a header line starting with "mass" or a comment line starting with #.
    """
    # Files named like: fe6fc4ee-..._MALDI1.txt
    # Key = filename without .txt extension = matches code column directly
    patterns = ['**/*.txt', '**/*.dat', '**/*.csv', '**/fid']
    found = {}
    for pat in patterns:
        for fpath in glob.glob(os.path.join(root, pat), recursive=True):
            basename = os.path.splitext(os.path.basename(fpath))[0]  # strip extension
            if re.match(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', basename):
                if basename not in found:
                    found[basename] = fpath
    print(f"[Spectra] Found {len(found)} unique files under {root}")
    if found:
        print(f"[Spectra] Example key: '{next(iter(found))}'")
    return found  # {code_without_ext: filepath}

def parse_spectrum(fpath):
    """
    Parse a spectrum file.
    Skips comment lines (#) and the header line.
    Returns (mass_array, intensity_array) as np.arrays, or (None, None) on failure.
    """
    masses, intensities = [], []
    try:
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('"'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    m, i = float(parts[0]), float(parts[1])
                    masses.append(m)
                    intensities.append(i)
                except ValueError:
                    continue
    except Exception as e:
        print(f"  [parse] Error reading {fpath}: {e}")
        return None, None

    if len(masses) < 10:
        return None, None
    return np.array(masses), np.array(intensities)

def extract_features(mass, intensity, mass_min=MASS_MIN, mass_max=MASS_MAX, n_bins=N_BINS):
    """
    Interpolate spectrum onto a common mass grid → fixed-length feature vector.
    Also computes summary statistics as additional features.
    """
    mask = (mass >= mass_min) & (mass <= mass_max)
    m, i = mass[mask], intensity[mask]
    if len(m) < 5:
        return None

    # Normalize intensity (TIC normalization)
    tic = np.trapz(i, m)
    if tic == 0:
        return None
    i = i / tic

    # Interpolate onto common grid
    grid = np.linspace(mass_min, mass_max, n_bins)
    try:
        interp = interp1d(m, i, kind='linear', bounds_error=False, fill_value=0)
        vec = interp(grid)
    except Exception:
        return None

    return vec  # shape: (n_bins,)

# ─────────────────────────────────────────────
# 3. BUILD DATASET
# ─────────────────────────────────────────────
def build_dataset(amr_df, spectra_map):
    """Match AMR rows to spectra by UUID, extract feature vectors."""
    records = []
    missing_spectra = 0

    for _, row in amr_df.iterrows():
        uuid = row['uuid']
        if uuid not in spectra_map:
            missing_spectra += 1
            continue

        fpath = spectra_map[uuid]
        mass, intensity = parse_spectrum(fpath)
        if mass is None:
            missing_spectra += 1
            continue

        vec = extract_features(mass, intensity)
        if vec is None:
            missing_spectra += 1
            continue

        rec = {'uuid': uuid, 'spectrum': vec}
        for drug in TARGET_DRUGS:
            if drug in row.index:
                rec[drug] = row[drug]
        records.append(rec)

    print(f"[Dataset] Matched: {len(records)} | Missing spectra: {missing_spectra}")
    return records

# ─────────────────────────────────────────────
# 4. STATISTICAL TESTS (per drug, per mass bin)
# ─────────────────────────────────────────────
def run_tests(records, drug):
    """
    For a given drug, split records into S vs R groups.
    Run Mann-Whitney U test at each mass bin.
    Returns: p_values array, u_statistics array, group sizes.
    """
    s_group = [r['spectrum'] for r in records if r.get(drug) == 'S']
    r_group = [r['spectrum'] for r in records if r.get(drug) == 'R']

    print(f"  [{drug}] S={len(s_group)}, R={len(r_group)}")

    if len(s_group) < 3 or len(r_group) < 3:
        print(f"  [{drug}] Too few samples, skipping.")
        return None

    s_mat = np.array(s_group)  # (n_S, n_bins)
    r_mat = np.array(r_group)  # (n_R, n_bins)

    n_bins = s_mat.shape[1]
    p_vals = np.zeros(n_bins)
    u_stats = np.zeros(n_bins)

    for b in range(n_bins):
        u, p = stats.mannwhitneyu(s_mat[:, b], r_mat[:, b], alternative='two-sided')
        p_vals[b] = p
        u_stats[b] = u

    # FDR correction (Benjamini-Hochberg)
    _, p_adj, _, _ = multipletests(p_vals, method='fdr_bh')

    return {
        'drug': drug,
        'n_S': len(s_group),
        'n_R': len(r_group),
        'p_raw': p_vals,
        'p_adj': p_adj,
        'u_stat': u_stats,
        's_mean': s_mat.mean(axis=0),
        'r_mean': r_mat.mean(axis=0),
    }

# ─────────────────────────────────────────────
# 5. VISUALISATION
# ─────────────────────────────────────────────
def plot_drug(result, mass_grid, outdir):
    drug = result['drug']
    safe_name = drug.replace('/', '-').replace(' ', '_')

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.suptitle(f"S. epidermidis — {drug}  (S n={result['n_S']}, R n={result['n_R']})",
                 fontsize=13, fontweight='bold')

    # Panel 1: Mean spectra
    ax = axes[0]
    ax.plot(mass_grid, result['s_mean'], color='#2ecc71', lw=1.2, label='S (sensitive)')
    ax.plot(mass_grid, result['r_mean'], color='#e74c3c', lw=1.2, label='R (resistant)')
    ax.set_ylabel('Mean TIC-normalised intensity')
    ax.legend(fontsize=9)
    ax.set_title('Mean spectra comparison')

    # Panel 2: -log10(p) raw
    ax = axes[1]
    log_p = -np.log10(result['p_raw'] + 1e-300)
    ax.fill_between(mass_grid, log_p, color='#3498db', alpha=0.6)
    ax.axhline(-np.log10(0.05), color='orange', lw=1, ls='--', label='p=0.05')
    ax.set_ylabel('-log₁₀(p) raw')
    ax.legend(fontsize=8)

    # Panel 3: -log10(p_adj) FDR
    ax = axes[2]
    log_padj = -np.log10(result['p_adj'] + 1e-300)
    ax.fill_between(mass_grid, log_padj, color='#9b59b6', alpha=0.6)
    ax.axhline(-np.log10(0.05), color='orange', lw=1, ls='--', label='FDR q=0.05')
    ax.set_ylabel('-log₁₀(q) FDR')
    ax.set_xlabel('Mass (m/z)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(outdir, f"{safe_name}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [Plot] saved → {out}")

def plot_summary(all_results, outdir):
    """Heatmap: fraction of mass bins with FDR q < 0.05 per drug."""
    drugs  = [r['drug'] for r in all_results]
    fracs  = [np.mean(r['p_adj'] < 0.05) for r in all_results]
    n_sig  = [np.sum(r['p_adj'] < 0.05) for r in all_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e74c3c' if f > 0.05 else '#3498db' for f in fracs]
    bars = ax.barh(drugs, fracs, color=colors)
    ax.axvline(0.05, color='orange', lw=1.5, ls='--', label='5% threshold')
    ax.set_xlabel('Fraction of significant mass bins (FDR q < 0.05)')
    ax.set_title('S. epidermidis — Spectral differences S vs R per drug')
    for bar, ns in zip(bars, n_sig):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{ns} bins', va='center', fontsize=8)
    ax.legend()
    plt.tight_layout()
    out = os.path.join(outdir, "summary_heatmap.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Summary plot] saved → {out}")

# ─────────────────────────────────────────────
# 6. SAVE RESULTS TABLE
# ─────────────────────────────────────────────
def save_results_table(all_results, outdir):
    rows = []
    for r in all_results:
        n_sig_raw = int(np.sum(r['p_raw'] < 0.05))
        n_sig_fdr = int(np.sum(r['p_adj'] < 0.05))
        min_p     = float(np.min(r['p_raw']))
        min_q     = float(np.min(r['p_adj']))
        rows.append({
            'Drug': r['drug'],
            'n_S': r['n_S'],
            'n_R': r['n_R'],
            'n_bins_p<0.05_raw': n_sig_raw,
            'n_bins_FDR<0.05': n_sig_fdr,
            'frac_bins_FDR<0.05': round(n_sig_fdr / len(r['p_adj']), 4),
            'min_p_raw': f"{min_p:.2e}",
            'min_q_FDR': f"{min_q:.2e}",
        })
    df = pd.DataFrame(rows)
    out = os.path.join(outdir, "results_summary.csv")
    df.to_csv(out, index=False)
    print(f"[Table] saved → {out}")
    print(df.to_string(index=False))
    return df


def save_significant_bins_table(all_results, mass_grid, outdir):
    """
    Detailed table: one row per significant bin (FDR q < 0.05).
    Columns: Drug, bin_index, mass_mz,
             S_mean_intensity, R_mean_intensity,
             higher_in, fold_diff_R_vs_S, p_raw, q_FDR
    """
    rows = []
    for r in all_results:
        sig_idx = np.where(r['p_adj'] < 0.05)[0]
        for i in sig_idx:
            s_mean  = float(r['s_mean'][i])
            rm_mean = float(r['r_mean'][i])
            if s_mean == 0 and rm_mean == 0:
                higher_in = 'equal'
                fold = float('nan')
            elif s_mean == 0:
                higher_in = 'R'
                fold = float('inf')
            elif rm_mean == 0:
                higher_in = 'S'
                fold = float('inf')
            else:
                fold = rm_mean / s_mean
                higher_in = 'R' if fold >= 1 else 'S'
                fold = fold if fold >= 1 else 1 / fold
            rows.append({
                'Drug':              r['drug'],
                'bin_index':         int(i),
                'mass_mz':           round(float(mass_grid[i]), 1),
                'S_mean_intensity':  f"{s_mean:.6e}",
                'R_mean_intensity':  f"{rm_mean:.6e}",
                'higher_in':         higher_in,
                'fold_diff_R_vs_S':  round(fold, 3) if not (isinstance(fold, float) and np.isinf(fold)) else 'inf',
                'p_raw':             f"{r['p_raw'][i]:.3e}",
                'q_FDR':             f"{r['p_adj'][i]:.3e}",
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[Significant bins] No significant bins found.")
        return df

    df = df.sort_values(['Drug', 'q_FDR']).reset_index(drop=True)
    out = os.path.join(outdir, "significant_bins_detail.csv")
    df.to_csv(out, index=False)
    print(f"[Significant bins table] saved → {out}  ({len(df)} rows)")

    for drug, grp in df.groupby('Drug'):
        print(f"\n  [{drug}]  {len(grp)} significant bins")
        print(grp[['mass_mz','higher_in','fold_diff_R_vs_S','p_raw','q_FDR']].to_string(index=False))
    return df


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Step 1: Load AMR data")
    amr_df = load_amr(AMR_CSV_PATH)

    print("\nStep 2: Find spectra files")
    spectra_map = find_spectra_files(SPECTRA_ROOT)

    print("\nStep 3: Build matched dataset")
    records = build_dataset(amr_df, spectra_map)

    if len(records) == 0:
        print("ERROR: No matched records found. Check paths and UUID formats.")
        return

    mass_grid = np.linspace(MASS_MIN, MASS_MAX, N_BINS)

    print("\nStep 4: Run Mann-Whitney U tests per drug")
    all_results = []
    for drug in TARGET_DRUGS:
        print(f"\n--- {drug} ---")
        result = run_tests(records, drug)
        if result is not None:
            all_results.append(result)
            plot_drug(result, mass_grid, OUTPUT_DIR)

    if not all_results:
        print("No results to report.")
        return

    print("\nStep 5: Summary")
    save_results_table(all_results, OUTPUT_DIR)
    save_significant_bins_table(all_results, mass_grid, OUTPUT_DIR)
    plot_summary(all_results, OUTPUT_DIR)

    print("\nDone! All outputs in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()