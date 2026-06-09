"""
Refactored Version
Methods:
1. Full Spectra
2. SHAP Top-k
3. Common Bins (MW ∩ SHAP)

Generated automatically.

AMR vs Mass Spectra Statistical Analysis & Machine Learning
Species: Staphylococcus epidermidis
Test: Mann-Whitney U test (S group vs R group) + Random Forest + SHAP
"""

import glob
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. CONFIGURATION — adjust paths here
# ─────────────────────────────────────────────
AMR_CSV_PATH    = r"C:\Users\鍾秉諺\Downloads\filtered_A_S_epidermidis\2018_S_epidermidis_metadata.csv"          
SPECTRA_ROOT    = r"C:\Users\鍾秉諺\Downloads\filtered_A_S_epidermidis\preprocessed"
OUTPUT_DIR      = "amr_spectra_results"
TARGET_SPECIES  = "Staphylococcus epidermidis"

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

# ML 實驗目標藥物
ML_TARGET_DRUGS = ["Ciprofloxacin", "Cotrimoxazole"]

MASS_MIN = 2000
MASS_MAX = 20000
N_BINS   = 500   

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD AMR DATA
# ─────────────────────────────────────────────

def split_data(X, y):
    
    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

def evaluate_method(
    X_train,
    X_test,
    y_train,
    y_test,
    selected_idx
):

    X_train_sel = X_train[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train_sel, y_train)

    prob = rf.predict_proba(
        X_test_sel
    )[:,1]

    auroc = roc_auc_score(
        y_test,
        prob
    )

    auprc = average_precision_score(
        y_test,
        prob
    )

    return (
        auroc,
        auprc,
        len(selected_idx)
    )

def load_amr(path):
    sep = '\t' if path.endswith('.tsv') else ','
    df = pd.read_csv(path, sep=sep, low_memory=False)
    df = df[df['species'].str.strip() == TARGET_SPECIES].copy()
    print(f"[AMR] {len(df)} rows for {TARGET_SPECIES}")
    
    df['uuid'] = df['code'].str.strip()
    df = df.dropna(subset=['uuid'])
    available_drugs = [d for d in TARGET_DRUGS if d in df.columns]
    
    return df[['uuid'] + available_drugs]

# ─────────────────────────────────────────────
# 2. FIND + PARSE SPECTRA FILES
# ─────────────────────────────────────────────
def find_spectra_files(root):
    patterns = ['**/*.txt', '**/*.dat', '**/*.csv', '**/fid']
    found = {}
    
    for pat in patterns:
        for fpath in glob.glob(os.path.join(root, pat), recursive=True):
            basename = os.path.splitext(os.path.basename(fpath))[0]  
            if re.match(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', basename):
                if basename not in found:
                    found[basename] = fpath
                    
    print(f"[Spectra] Found {len(found)} unique files under {root}")
    return found

def parse_spectrum(fpath):
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
    except Exception:
        return None, None
        
    if len(masses) < 10:
        return None, None
        
    return np.array(masses), np.array(intensities)

def extract_features(mass, intensity, mass_min=MASS_MIN, mass_max=MASS_MAX, n_bins=N_BINS):
    mask = (mass >= mass_min) & (mass <= mass_max)
    m, i = mass[mask], intensity[mask]
    
    if len(m) < 5:
        return None
        
    tic = np.trapz(i, m)
    if tic == 0:
        return None
        
    i = i / tic
    grid = np.linspace(mass_min, mass_max, n_bins)
    
    try:
        interp = interp1d(m, i, kind='linear', bounds_error=False, fill_value=0)
        vec = interp(grid)
    except Exception:
        return None
        
    return vec

# ─────────────────────────────────────────────
# 3. BUILD DATASET
# ─────────────────────────────────────────────
def build_dataset(amr_df, spectra_map):
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
# 4. STATISTICAL TESTS 
# ─────────────────────────────────────────────
def run_tests(records, drug):
    s_group = [r['spectrum'] for r in records if r.get(drug) == 'S']
    r_group = [r['spectrum'] for r in records if r.get(drug) == 'R']
    
    if len(s_group) < 3 or len(r_group) < 3:
        return None
        
    s_mat = np.array(s_group)
    r_mat = np.array(r_group)
    n_bins = s_mat.shape[1]
    
    p_vals = np.zeros(n_bins)
    u_stats = np.zeros(n_bins)
    effect_sizes = np.zeros(n_bins)

    n1 = len(s_group)
    n2 = len(r_group)

    for b in range(n_bins):
        u, p = stats.mannwhitneyu(s_mat[:, b], r_mat[:, b], alternative='two-sided')
        p_vals[b] = p
        u_stats[b] = u

        # Cliff's delta
        delta = (2 * u) / (n1 * n2) - 1
        effect_sizes[b] = delta
        
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
        'effect_size': effect_sizes,
    }

# ─────────────────────────────────────────────
# 5. SAVE STATS TABLES
# ─────────────────────────────────────────────
def save_significant_bins_table(all_results, mass_grid, outdir):
    rows = []
    for r in all_results:
        sig_idx = np.where(r['p_adj'] < 0.05)[0]
        for i in sig_idx:
            s_mean = float(r['s_mean'][i])
            rm_mean = float(r['r_mean'][i])
            higher_in = 'R' if rm_mean > s_mean else 'S'
            
            fold = (rm_mean / s_mean) if s_mean != 0 else float('inf')
            fold = fold if fold >= 1 else (1 / fold if fold != 0 else float('inf'))
            
            rows.append({
                'Drug': r['drug'], 
                'bin_index': int(i), 
                'mass_mz': round(float(mass_grid[i]), 1),
                'q_FDR': r['p_adj'][i],
                'effect_size': round(r['effect_size'][i], 4),
                'higher_in': higher_in,
                'S_mean': round(s_mean, 6),
                'R_mean': round(rm_mean, 6),
                'fold_change': round(fold, 3)
            })
            
    df = pd.DataFrame(rows)
    if not df.empty:
        out = os.path.join(outdir, "significant_bins_detail.csv")
        df.to_csv(out, index=False)
    return df

# ─────────────────────────────────────────────
# 6. MACHINE LEARNING & SHAP MODULE
# ─────────────────────────────────────────────
def build_ml_dataset(records, drug):
    X, y = [], []
    for r in records:
        if drug in r and r[drug] in ['S', 'R']:
            X.append(r['spectrum'])
            y.append(0 if r[drug] == 'S' else 1)
    return np.array(X), np.array(y)

def select_features_mw(X_train, y_train, alpha=0.05):
    s_mat = X_train[y_train == 0]
    r_mat = X_train[y_train == 1]
    n_features = X_train.shape[1]
    p_vals = np.zeros(n_features)

    for i in range(n_features):
        _, p = stats.mannwhitneyu(s_mat[:, i], r_mat[:, i], alternative='two-sided')
        p_vals[i] = p

    _, q_vals, _, _ = multipletests(p_vals, method='fdr_bh')
    selected_idx = np.where(q_vals < alpha)[0]
    return selected_idx

def select_features_shap(X_train, y_train, top_k):
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_train)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    mean_importance = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_importance)[::-1][:top_k]
    return top_idx



def run_four_experiments(records):
    
    results = []

    for drug in ML_TARGET_DRUGS:

        print("\n" + "="*60)
        print(drug)

        X, y = build_ml_dataset(
            records,
            drug
        )

        X_train, X_test, y_train, y_test = split_data(X,y)

        # ==================================================
        # Method 1
        # ==================================================

        idx_full = np.arange(
            X.shape[1]
        )

        auroc, auprc, n_feat = evaluate_method(
            X_train,
            X_test,
            y_train,
            y_test,
            idx_full
        )

        results.append({
            "Drug":drug,
            "Method":"Full Spectra",
            "Features":n_feat,
            "AUROC":round(auroc,4),
            "AUPRC":round(auprc,4)
        })

        # ==================================================
        # Method 2
        # ==================================================

        mw_idx = select_features_mw(
            X_train,
            y_train
        )

        auroc, auprc, n_feat = evaluate_method(
            X_train,
            X_test,
            y_train,
            y_test,
            mw_idx
        )

        results.append({
            "Drug":drug,
            "Method":"MW+FDR",
            "Features":n_feat,
            "AUROC":round(auroc,4),
            "AUPRC":round(auprc,4)
        })

        # ==================================================
        # Method 3
        # ==================================================

        shap_idx = select_features_shap(
            X_train,
            y_train,
            top_k=len(mw_idx)
        )

        auroc, auprc, n_feat = evaluate_method(
            X_train,
            X_test,
            y_train,
            y_test,
            shap_idx
        )

        results.append({
            "Drug":drug,
            "Method":"SHAP Top-k",
            "Features":n_feat,
            "AUROC":round(auroc,4),
            "AUPRC":round(auprc,4)
        })

        # ==================================================
        # Method 4
        # ==================================================

        common_idx = np.intersect1d(
            mw_idx,
            shap_idx
        )

        auroc, auprc, n_feat = evaluate_method(
            X_train,
            X_test,
            y_train,
            y_test,
            common_idx
        )

        results.append({
            "Drug":drug,
            "Method":"Common Bins",
            "Features":n_feat,
            "AUROC":round(auroc,4),
            "AUPRC":round(auprc,4)
        })

    df = pd.DataFrame(results)

    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "four_method_results.csv"
        ),
        index=False
    )

    print(df)

    return df


def final_shap_plot(records, drug):
    X, y = build_ml_dataset(records, drug)

    X_train, X_test, y_train, y_test = split_data(X,y)
    mw_idx = select_features_mw(X_train,y_train)
    k = len(mw_idx)
    # ---------- First RF ----------
    rf_full = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_full.fit(X_train,y_train)

    explainer_full = shap.TreeExplainer(rf_full)
    shap_values_full = explainer_full.shap_values(X_train)

    # SHAP compatibility
    if isinstance(shap_values_full, list):
        shap_values_full = shap_values_full[1]
    elif len(shap_values_full.shape) == 3:
        shap_values_full = shap_values_full[:, :, 1]

    # Number of features determined by MW+FDR

    print(f"{drug}: SHAP Top-{k}")

    mean_importance = np.abs(shap_values_full).mean(axis=0)
    top_idx = np.argsort(mean_importance)[::-1][:k]

    # ---------- Second RF ----------
    X_top = X_train[:, top_idx]

    rf_top = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_top.fit(X_top, y_train)

    explainer_top = shap.TreeExplainer(rf_top)
    shap_values_top = explainer_top.shap_values(X_top)

    # SHAP compatibility
    if isinstance(shap_values_top, list):
        shap_values_top = shap_values_top[1]
    elif len(shap_values_top.shape) == 3:
        shap_values_top = shap_values_top[:, :, 1]

    mass_grid = np.linspace(MASS_MIN, MASS_MAX, N_BINS)
    feature_names = [f"{mass_grid[i]:.0f} Da" for i in top_idx]

    # ---------- Plot ----------
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_top,
        X_top,
        feature_names=feature_names,
        show=False,
        max_display=10,      # Top 10 peaks for readability
        plot_size=None       # IMPORTANT!
    )

    plt.title(
        f"SHAP Top-{k} Summary ({drug})",
        fontsize=18,
        fontweight='bold',
        pad=20
    )

    plt.xlabel("SHAP value (impact on model output)", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"SHAP_Top{k}_{drug}.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    return top_idx
    
def final_shap_plot_mw(records, drug):
    X, y = build_ml_dataset(records, drug)

    # ===== MW selected features =====
    X_train, X_test, y_train, y_test = split_data(X,y)

    mw_idx = select_features_mw(X_train,y_train)
    print(f"{drug}: MW features = {len(mw_idx)}")

    X_mw = X_train[:, mw_idx]

    # ===== RF =====
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_mw,y_train)

    # ===== SHAP =====
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_mw)

    # compatibility
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    # feature names
    mass_grid = np.linspace(MASS_MIN, MASS_MAX, N_BINS)
    feature_names = [f"{mass_grid[i]:.0f} Da" for i in mw_idx]

    # ===== Plot =====
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_mw,
        feature_names=feature_names,
        show=False,
        max_display=10,
        plot_size=None
    )

    plt.title(
        f"MW+FDR SHAP Summary ({drug})",
        fontsize=18,
        fontweight='bold',
        pad=20
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"MW_SHAP_{drug}.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    return mw_idx

def final_shap_plot_common(records, drug):
    
    X, y = build_ml_dataset(records, drug)

    # =====================================
    # MW selected bins
    # =====================================
    X_train, X_test, y_train, y_test = split_data(X,y)

    mw_idx = select_features_mw(X_train, y_train)

    # ===================================== 
    # SHAP selected bins
    # =====================================
    rf_full = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    rf_full.fit( X_train,y_train)

    explainer_full = shap.TreeExplainer(rf_full)
    shap_values_full = explainer_full.shap_values(X_train)

    # SHAP compatibility
    if isinstance(shap_values_full, list):
        shap_values_full = shap_values_full[1]
    elif len(shap_values_full.shape) == 3:
        shap_values_full = shap_values_full[:, :, 1]

    mean_importance = np.abs(
        shap_values_full
    ).mean(axis=0)

    k = len(mw_idx)

    shap_idx = np.argsort(
        mean_importance
    )[::-1][:k]

    # =====================================
    # Common bins
    # =====================================
    common_idx = np.intersect1d(
        mw_idx,
        shap_idx
    )

    print("\n" + "="*60)
    print(drug)
    print(f"MW bins      : {len(mw_idx)}")
    print(f"SHAP bins    : {len(shap_idx)}")
    print(f"Common bins  : {len(common_idx)}")

    if len(common_idx) == 0:
        print("No common bins found.")
        return None

    # =====================================
    # RF using common bins
    # =====================================
    X_common = X_train[:, common_idx]

    rf_common = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    rf_common.fit(
        X_common,
        y_train
    )

    explainer_common = shap.TreeExplainer(
        rf_common
    )

    shap_values_common = explainer_common.shap_values(
        X_common
    )

    # SHAP compatibility
    if isinstance(shap_values_common, list):
        shap_values_common = shap_values_common[1]
    elif len(shap_values_common.shape) == 3:
        shap_values_common = shap_values_common[:, :, 1]

    # =====================================
    # Feature names
    # =====================================
    mass_grid = np.linspace(
        MASS_MIN,
        MASS_MAX,
        N_BINS
    )

    feature_names = [
        f"{mass_grid[i]:.0f} Da"
        for i in common_idx
    ]

    # =====================================
    # Save common bins table
    # =====================================
    pd.DataFrame({
        "bin_index": common_idx,
        "mass_mz": [
            round(mass_grid[i], 1)
            for i in common_idx
        ]
    }).to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"Common_Bins_{drug}.csv"
        ),
        index=False
    )

    # =====================================
    # SHAP Plot
    # =====================================
    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values_common,
        X_common,
        feature_names=feature_names,
        show=False,
        max_display=min(
            10,
            len(common_idx)
        ),
        plot_size=None
    )

    plt.title(
        f"Common Bins SHAP Summary ({drug})",
        fontsize=18,
        fontweight='bold',
        pad=20
    )

    plt.xlabel(
        "SHAP value (impact on model output)",
        fontsize=12
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"Common_SHAP_{drug}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return common_idx

def compare_common_peaks(records, drug):
    X, y = build_ml_dataset(records, drug)

    # MW selected peaks
    mw_idx = select_features_mw(X, y)

    # SHAP selected peaks
    shap_idx = final_shap_plot(records, drug)

    # Intersection
    common_idx = np.intersect1d(mw_idx, shap_idx)

    mass_grid = np.linspace(MASS_MIN, MASS_MAX, N_BINS)
    common_peaks = [round(mass_grid[i], 1) for i in common_idx]

    print("\n==============================")
    print(drug)
    print("==============================")
    print(f"MW features   : {len(mw_idx)}")
    print(f"SHAP features : {len(shap_idx)}")
    print(f"Common peaks  : {len(common_idx)}")
    print("\nCommon m/z peaks:")

    for p in common_peaks:
        print(f"{p} Da")
        
    df = pd.DataFrame({"mass_mz": common_peaks})
    df.to_csv(os.path.join(OUTPUT_DIR, f"Common_Peaks_{drug}.csv"), index=False)
    
    return common_idx

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
        return

    mass_grid = np.linspace(MASS_MIN, MASS_MAX, N_BINS)

    print("\nStep 4: Run Mann-Whitney U tests per drug")
    all_results = []
    
    for drug in TARGET_DRUGS:
        result = run_tests(records, drug)
        if result is not None:
            all_results.append(result)

    print("\nStep 5: Save Statistical Summary")
    sig_df = save_significant_bins_table(all_results, mass_grid, OUTPUT_DIR)
    
    print("\nStep 6: Four Experiments")

    run_four_experiments(records)

    print("\nGenerate SHAP plots")

    for drug in ML_TARGET_DRUGS:

        final_shap_plot_mw(
            records,
            drug
        )

        final_shap_plot(
            records,
            drug
        )

        final_shap_plot_common(
            records,
            drug
        )
    print("\nDone! All outputs in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()